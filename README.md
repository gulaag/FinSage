# FinSage — SEC申告書インテリジェンスプラットフォーム

> **Databricks上の本番グレード・メダリオンデータパイプライン**、GitHub ActionsによるフルCI/CDを備えたDatabricksアセットバンドル（DAB）としてデプロイされます。

FinSageは、30社の大型株企業の年次（10-K）および四半期（10-Q）SEC申告書を取り込み、XBRL財務指標を正規化し、ナラティブセクションを抽出し、検索拡張生成（RAG）向けのトークンベースのベクターインデックスを公開します。

> **英語版READMEは [README_english.md](README_english.md) を参照してください。**

---

## 目次

1. [アーキテクチャ概要](#1-アーキテクチャ概要)
2. [メダリオン層](#2-メダリオン層)
3. [リポジトリ構成](#3-リポジトリ構成)
4. [Databricksアセットバンドル](#4-databricksアセットバンドル)
5. [CI/CD — GitHub Actions](#5-cicd--github-actions)
6. [ブランチ戦略](#6-ブランチ戦略)
7. [ローカル開発](#7-ローカル開発)
8. [テストの実行](#8-テストの実行)
9. [デプロイメント参照](#9-デプロイメント参照)
10. [環境変数とシークレット](#10-環境変数とシークレット)
11. [ノートブックコード詳細解説](#11-ノートブックコード詳細解説)

---

## 1. アーキテクチャ概要

```
                       ┌──────────────────────────────────────────────────┐
                       │               GitHubリポジトリ                    │
                       │   feature/* ──► dev ──► main                     │
                       └──────────────────────┬───────────────────────────┘
                                              │  mainへのpush
                                              ▼
                       ┌──────────────────────────────────────────────────┐
                       │         GitHub Actionsワークフロー               │
                       │  1. pytest（ユニットテスト）                      │
                       │  2. databricks bundle validate                   │
                       │  3. databricks bundle deploy -t prod             │
                       └──────────────────────┬───────────────────────────┘
                                              │  デプロイ
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Databricksワークスペース（本番）                         │
│                                                                             │
│  SEC EDGAR / SEC API ──► [01 スキーマセットアップ]                           │
│                               │                                             │
│                               ▼                                             │
│                         [02 ブロンズ]  ◄── Auto Loader (cloudFiles)         │
│                               │          + CompanyFacts API                 │
│                               ▼                                             │
│                         [03 シルバー] ◄── XBRLフラット化 + セクションNLP     │
│                               │                                             │
│                               ▼                                             │
│                         [04 ゴールド] ◄── 指標集計 + 前年比成長率            │
│                               │                                             │
│                               ▼                                             │
│                         [05 ベクター] ◄── tiktokenチャンク化 + VSインデックス │
│                                                                             │
│   すべての層はUnity Catalog（mainカタログ）内のDelta Lakeテーブルに保存       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. メダリオン層

### ブロンズ層 — 生データ取り込み（`main.finsage_bronze`）

ブロンズ層は**追記専用かつ監査可能**です。ビジネスロジックは適用されません。

| テーブル | 内容 |
|---|---|
| `filings` | すべてのSEC申告書ファイルの生バイナリデータ。Databricks Auto Loader（`cloudFiles`フォーマット）で取り込まれます。Auto Loaderのチェックポイントによってexactly-onceデリバリーが保証されます。 |
| `xbrl_companyfacts_raw` | SEC EDGAR CompanyFacts API（`/api/xbrl/companyfacts/CIK{cik}.json`）からの生JSONペイロード。ティッカーごと・日ごとに1行。 |
| `ingestion_errors` | すべての層からのダウンロード失敗、HTTPエラー、パース例外をログに記録。可観測性と再処理のために使用。 |
| `sec_filings_download_log` | `sec-edgar-downloader`並列ダウンロードジョブの冪等性ログ。再実行時の重複API呼び出しを防止。 |

**主要設計方針：**
- すべてのブロンズテーブルで`delta.enableChangeDataFeed = true`を有効化し、ダウンストリームのChange Data Captureを実現。
- Auto Loaderの`availableNow=True`トリガーにより、ストリームをバッチとして動作させます（新しいファイルをすべて処理してから停止）。

### シルバー層 — クリーニング・解析済み（`main.finsage_silver`）

シルバー層はブロンズデータに対して2つの独立した変換を適用します：

| テーブル | 変換 |
|---|---|
| `financial_statements` | XBRL CompanyFacts JSONを`TARGET_CONCEPT_MAP`を使用してフラット化。30以上の生XBRLコンセプトを11の正規化指標名（`revenue`、`net_income`など）に正規マッピング。SHA-256の`statement_id`で重複排除。毎回実行時に冪等なMERGE。 |
| `filing_sections` | 10-KのHTMLファイルバイトをデコードし、HTMLタグ・Base64画像・スクリプトを除去後、正規表現による境界検出を使用して3つの名前付きセクション（**事業概要（Item 1）**、**リスク要因（Item 1A）**、**MD&A（Item 7）**）に解析。セクション境界は決定論的かつ監査可能。 |

**主要設計方針：**
- `TARGET_CONCEPT_MAP`は指標正規化の唯一の真実のソースであり、Sparkとは独立してユニットテスト済み（`tests/unit/test_normalizer.py`参照）。
- セクション抽出では`10-K`申告書のみを処理 — 四半期報告書にはItem 1/7構造がありません。

### ゴールド層 — 分析用指標（`main.finsage_gold`）

ゴールド層は、ティッカー・年度ごとに派生KPIを含む**ワイドで分析レディな**テーブルを生成します。

| テーブル | 内容 |
|---|---|
| `company_metrics` | `(ticker, fiscal_year)`ごとに1行。16の財務指標、粗利益率%、売上高前年比成長率%、負債資本比率、9つのコア指標のうち何個が入力されたかを示す`data_quality_score`（0〜1）を含む。 |
| `filing_section_chunks` | シルバーセクションのトークンベースのチャンク（512トークン、64トークンオーバーラップ）。冪等なマージのための決定論的SHA-256 `chunk_id`付き。 |

**主要設計方針：**
- 厳密な会計期間の整合：フロー指標には`fiscal_period = 'FY'`のみを使用し、`duration_days`は350〜380でなければなりません。
- 正規のアクセッション番号選択：集計前に必須指標のカバレッジに基づいて`(ticker, fiscal_year)`ごとに1つの`accession_number`が選ばれます。

### ベクター層 — RAGインデックス（`main.finsage_gold`）

| リソース | 内容 |
|---|---|
| `filing_section_chunks`（ゴールド） | `delta.enableChangeDataFeed = true`のソーステーブル。 |
| `filing_chunks_index` | `databricks-bge-large-en`でバックされたDatabricks Vector Search Delta Syncインデックス。ダウンストリームRAGエージェントの類似検索をサポート。 |

---

## 3. リポジトリ構成

```
FinSage/
├── databricks.yml                     # Databricksアセットバンドルルート設定
├── databricks/
│   ├── notebooks/
│   │   ├── 01_schema_setup.py         # DDL：スキーマ、テーブル、ボリューム + SEC申告書ダウンロード
│   │   ├── 02_bronze_autoloader.py    # Auto Loader + SEC API取り込み
│   │   ├── 03_silver_decoder.py       # XBRLフラット化 + セクション抽出
│   │   ├── 04_gold_metrics.py         # 指標集計 + KPI導出
│   │   └── 05_vector_chunker.py       # チャンク化 + Vector Searchセットアップ
│   └── workflows/                     # （将来のワークフローYAML用に予約）
├── terraform/
│   └── main.tf                        # クラスターポリシー、シークレットスコープ、SPルックアップ
├── .github/
│   └── workflows/
│       └── deploy.yml                 # CI/CDパイプライン（pytest → validate → deploy）
├── tests/
│   └── unit/
│       └── test_normalizer.py         # pytest：TARGET_CONCEPT_MAPカバレッジ
├── assets/
│   └── screenshots/
│       └── finsage_dag_databricks.png # DatabricksワークスペースのライブDAGビュー
├── src/
│   ├── ingestion/
│   ├── processing/
│   ├── retrieval/
│   ├── serving/
│   └── agent/
├── docs/
│   ├── FinSage_blueprint_v2.html
│   ├── challenges_log.html
│   └── technical_decisions.html
├── requirements.txt
├── README.md                          # 本ファイル（日本語版）
└── README_english.md                  # 英語版README
```

> **Databricksノートブックソースファイル：** `databricks/notebooks/`配下のすべての`.py`ファイルは`# Databricks notebook source`で始まります。DABデプロイステップによってDatabricksワークスペースにアップロードされると、Databricksはそれらをインタラクティブなノートブックとして認識しますが、Gitでは通常のPythonファイルのまま残ります。

---

## 4. Databricksアセットバンドル

FinSageは**Databricksアセットバンドル（DAB）**としてデプロイされます。クラスター、タスク、スケジュール、環境プロモーションを含むジョブトポロジー全体がリポジトリルートの`databricks.yml`に定義されています。

### バンドル構成

```yaml
bundle:
  name: finsage_pipeline

resources:
  jobs:
    finsage_daily_run:
      tasks:
        - task_key: schema_setup        # 01_schema_setup.py
        - task_key: bronze_autoloader   # 02_bronze_autoloader.py  (depends_on: schema_setup)
        - task_key: silver_decoder      # 03_silver_decoder.py     (depends_on: bronze_autoloader)
        - task_key: gold_metrics        # 04_gold_metrics.py       (depends_on: silver_decoder)
        - task_key: vector_chunker      # 05_vector_chunker.py     (depends_on: gold_metrics)
```

タスクは`depends_on`により**厳密に順次実行**されます。いずれかのタスクが失敗すると、すべてのダウンストリームタスクがスキップされ、メールアラートが送信されます。

### ターゲット

| ターゲット | モード | 目的 |
|---|---|---|
| `dev`（デフォルト） | `development` | 個人デプロイ；ジョブ名にプレフィックス付き；安全にイテレーション可能。 |
| `prod` | `production` | 共有デプロイ；名前プレフィックスなし；`main`へのCI/CDにより起動。 |

### CLIコマンド

```bash
# バンドルの検証（構文 + ワークスペース接続チェック）
databricks bundle validate

# devへのデプロイ（デフォルトターゲット）
databricks bundle deploy

# 特定ターゲットへのデプロイ
databricks bundle deploy -t prod

# devでジョブを手動実行
databricks bundle run finsage_daily_run

# prodでジョブを手動実行
databricks bundle run -t prod finsage_daily_run

# デプロイ済みリソースを破棄（devのみ — 本番では承認なしに実行しないこと）
databricks bundle destroy
```

### ライブDAG — Databricksワークスペース

以下のスクリーンショットは、`databricks bundle deploy`成功後にDatabricks Jobs & Pipelines UIに表示される`finsage_daily_run`ジョブです。5つのタスクすべてが厳密に順次実行されるDAGとして接続され、タスク間のコールドスタートのオーバーヘッドを避けるために`finsage_cluster`ジョブクラスターを共有しています。

<img src="assets/screenshots/pipeline.gif" alt="DatabricksワークスペースのFinSageパイプラインDAG — schema_setup → bronze_autoloader → silver_decoder → gold_metrics → vector_chunkerの5つの順次タスク、すべて共有finsage_clusterで実行。UTC 06:00に毎日スケジュール。" style="max-width:100%;" />

> 開発モードではジョブは**[dev Digvijay]**とマークされます — DABは自動的にデプロイユーザー名をジョブ名にプレフィックスとして付け、同じ共有ワークスペース内の本番`finsage_daily_run`ジョブとの衝突を防ぎます。

---

## 5. CI/CD — GitHub Actions

ワークフローファイルは`.github/workflows/deploy.yml`です。

### パイプラインステージ

```
mainへのpush
     │
     ▼
┌────────────────┐     失敗     ┌─────────────────────────────────────────┐
│  unit-tests    │────────────►│  パイプライン停止。デプロイは行われない。  │
│  (pytest)      │              └─────────────────────────────────────────┘
└───────┬────────┘
        │ 成功
        ▼
┌────────────────────────┐
│  bundle-validate       │  databricks bundle validate
│  (Databricks CLI)      │
└────────────┬───────────┘
             │ 成功
             ▼
┌────────────────────────┐
│  deploy-prod           │  databricks bundle deploy -t prod
│  (Databricks CLI)      │
└────────────────────────┘
```

### ワークフローの動作

| トリガー | unit-tests | bundle-validate | deploy-prod |
|---|---|---|---|
| `main`へのpush | ✓ | ✓ | ✓ |
| `main`へのプルリクエスト | ✓ | ✗ | ✗ |
| `dev`またはフィーチャーブランチへのpush | ✗ | ✗ | ✗ |

### 認証

CLIはサービスプリンシパルを介した**OAuthマシン間（M2M）**認証を使用します。このワークスペースではエンタープライズポリシーにより従来のパーソナルアクセストークン（PAT）は無効化されています。

| シークレット | 値 |
|---|---|
| `DATABRICKS_HOST` | `https://<your-workspace>.cloud.databricks.com` |
| `DATABRICKS_CLIENT_ID` | `finsage-service-principal`のアプリケーション（クライアント）ID |
| `DATABRICKS_CLIENT_SECRET` | `finsage-service-principal`のOAuthシークレット |

GitHubリポジトリの**Settings → Secrets and variables → Actions**でこれらを追加してください。

サービスプリンシパルのプロビジョニング（Databricks管理者アクセスが必要）：
1. IDプロバイダー（Entra ID / Okta）でサービスプリンシパルを作成。
2. SPにDatabricksワークスペースの**Can Manage**ロールを付与。
3. SPのOAuthシークレットを生成。
4. 上記3つのシークレットをGitHubに追加。

---

## 6. ブランチ戦略

```
main          ──── 保護済み；PR + CI通過が必要 ──────────────────────────►
                         ▲                    ▲
                         │ マージ              │ マージ
dev           ──── 統合テスト ────────────────┘
                         ▲
                         │ マージ
feature/*     ──── 個別フィーチャー作業 ────────────────────────────────►
```

| ブランチ | 目的 | デプロイ先 |
|---|---|---|
| `feature/*` | 新機能、バグ修正。短命。 | なし（PRではCIユニットテストのみ） |
| `dev` | 統合テスト；ステージング相当。 | Databricks`dev`ターゲット（手動`bundle deploy`） |
| `main` | 本番対応コード。PRのみでマージ。 | Databricks`prod`ターゲット（GitHub Actions自動化） |

### リリースプロセス

1. `main`から`feature/my-change`ブランチを作成。
2. ローカルで開発・テスト（§7参照）。
3. `main`へのプルリクエストを開く。GitHub Actionsが自動的にユニットテストを実行。
4. PRが承認されCIが通過したら、`main`にマージ。
5. CI/CDパイプラインが自動的に検証して`prod`にデプロイ。

> **ホットフィックス：** `main`から直接ブランチを切り、修正を適用してPRを開いてください。PRプロセスを迂回しないでください — `bundle validate`ゲートは本番前の最後の防衛線です。

---

## 7. ローカル開発

### 前提条件

- Python 3.11以上
- [Databricks CLI v0.218以上](https://docs.databricks.com/dev-tools/cli/databricks-cli.html)
- Unity Catalogが有効化されたDatabricksワークスペース

### セットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/<your-org>/FinSage.git
cd FinSage

# 2. 仮想環境を作成
python -m venv .venv
source .venv/bin/activate

# 3. 依存関係をインストール
pip install -r requirements.txt

# 4. OAuth U2M（ブラウザベースログイン）でDatabricks CLIを設定
databricks auth login --host https://dbc-f33010ed-00fc.cloud.databricks.com/
# ブラウザが開き、一度だけログインが必要。PATは不要。
# CI環境では、DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET（M2M OAuth）で認証。

# 5. 個人dev環境にデプロイ
databricks bundle deploy       # デフォルトで'dev'ターゲットにデプロイ

# 6. 手動実行をトリガー
databricks bundle run finsage_daily_run
```

---

## 8. テストの実行

テストは`tests/unit/`に配置されており、**Spark依存なし** — 標準ライブラリとpytestのみを使用したプレーンPythonで実行されます。

```bash
# 全ユニットテストを実行
pytest tests/unit/ -v

# カバレッジレポート付きで実行
pytest tests/unit/ -v --cov=databricks/notebooks --cov-report=term-missing

# 単一テストファイルを実行
pytest tests/unit/test_normalizer.py -v
```

### テスト対象

| テストファイル | カバレッジ内容 |
|---|---|
| `test_normalizer.py` | `03_silver_decoder.py`の`TARGET_CONCEPT_MAP`。すべてのXBRLコンセプトが期待される正規化指標名に解決されること、未知のコンセプトが`None`を返すこと、マップが構造的に完全であることをアサート。 |

---

## 9. デプロイメント参照

### 初回セットアップ

```bash
# バンドルを検証（YAML + ワークスペース権限チェック）
databricks bundle validate

# devにデプロイ
databricks bundle deploy

# prodにデプロイ（通常はCIが行うが、手動でも実行可能）
databricks bundle deploy -t prod
```

### 既存デプロイメントの更新

```bash
# ノートブックまたはdatabricks.yml変更後：
git add .
git commit -m "feat: update silver section extraction regex"
git push origin main
# → GitHub Actionsが残りを処理
```

### 実行中ジョブの監視

```bash
# 最近のジョブ実行を一覧表示
databricks jobs list-runs --job-id <job-id>

# 特定の実行のログを表示
databricks runs get-output --run-id <run-id>
```

---

## 10. 環境変数とシークレット

| 名前 | 設定場所 | 目的 |
|---|---|---|
| `DATABRICKS_HOST` | GitHubシークレット | CIでのCLI認証用ワークスペースURL |
| `DATABRICKS_CLIENT_ID` | GitHubシークレット | CIでのM2M OAuthのサービスプリンシパルクライアントID |
| `DATABRICKS_CLIENT_SECRET` | GitHubシークレット | CIでのM2M認証のサービスプリンシパルOAuthシークレット |
| `USER_AGENT` | ノートブックウィジェットデフォルト | SEC EDGAR APIへのFinSageの識別（SEC利用規約で必須） |

> **シークレットをGitにコミットしないでください。** `.gitignore`はすでに`.env`ファイルを除外しています。CIにはGitHubシークレットを使用し、ノートブック内のランタイムシークレットにはDatabricksシークレットスコープ（`databricks secrets`）を使用してください。

---

## 11. ノートブックコード詳細解説

このセクションでは、5つのDatabricksノートブックそれぞれのコードを詳細に解説します。各ノートブックはメダリオンアーキテクチャの特定の役割を担い、SEC申告書データが生データから分析可能なベクターインデックスへと変換されるまでの流れを構成しています。

---

### ノートブック 01 — `01_schema_setup.py`：スキーマ初期化 & SEC申告書ダウンロード

このノートブックはパイプライン全体の起点です。インフラを構築し、30社分のSEC申告書を並列でダウンロードします。何度実行しても安全な**冪等設計**となっています。

#### セクション1：ランタイムパラメータ設定

```python
dbutils.widgets.text("catalog",       "main",       "Unity Catalog catalog")
dbutils.widgets.text("env",           "dev",        "Environment (dev/prod)")
dbutils.widgets.text("start_date",    "2020-01-01", "Earliest filing date")
dbutils.widgets.text("ticker_filter", "",           "Comma-separated tickers (empty=all)")
```

`dbutils.widgets`はDatabricksのパラメータ注入メカニズムです。DABジョブが`databricks.yml`の`base_parameters`を通じてこれらの値を渡します。インタラクティブな実行の場合はデフォルト値が使用されます。

- `catalog`：Unity Catalogのカタログ名（例：`main`）
- `env`：実行環境。`dev`と`prod`で動作を切り替えます
- `start_date`：`2020-01-01`以降の申告書のみを取り込む起算日
- `ticker_filter`：処理対象のティッカーをカンマ区切りで指定。空の場合は全30社が対象

```python
TICKER_SUBSET = [t.strip() for t in TICKER_FILTER.split(",") if t.strip()] if TICKER_FILTER else []
```

このリスト内包表記は`"AAPL, MSFT, GOOGL"`という文字列を`["AAPL", "MSFT", "GOOGL"]`というリストに変換します。空文字を除去するため`t.strip()`と`if t.strip()`の二重チェックを行っています。

#### セクション2：メダリオンスキーマの作成（SQL）

```sql
CREATE SCHEMA IF NOT EXISTS main.finsage_bronze;
CREATE SCHEMA IF NOT EXISTS main.finsage_silver;
CREATE SCHEMA IF NOT EXISTS main.finsage_gold;
```

`IF NOT EXISTS`によって何度実行しても安全です。3つのスキーマはデータ品質の3層（生データ→クリーニング済み→分析用）を表します。

#### セクション3：ダウンロードログテーブルの作成（SQL）

```sql
CREATE TABLE IF NOT EXISTS main.finsage_bronze.sec_filings_download_log (
    ticker            STRING,
    form_type         STRING,
    last_successful_run DATE,
    status            STRING,
    retry_count       INT,
    error_message     STRING,
    updated_at        TIMESTAMP
) USING DELTA;
```

このテーブルはジョブ実行間の冪等性を保証するための状態管理テーブルです。同じ日に同じ（ticker, form_type）の組み合わせを二重ダウンロードしないよう制御します。

#### セクション4：ボリュームの作成

```sql
CREATE VOLUME IF NOT EXISTS main.finsage_bronze.raw_filings;
```

Unity Catalogのボリュームはオブジェクトストレージ上のファイルシステムパスです。SEC申告書の生HTMLファイルはここに保存されます。アクセスパスは`/Volumes/main/finsage_bronze/raw_filings/`となります。

#### セクション5：設定定数と30社ティッカーリスト

```python
VOLUME_PATH = f"/Volumes/{CATALOG}/finsage_bronze/raw_filings"
USER_AGENT  = "Arsaga Partners digvijay@arsaga.jp"
LOG_TABLE   = f"{CATALOG}.finsage_bronze.sec_filings_download_log"
MAX_RETRIES = 3
MAX_CONCURRENT_WORKERS = 3

_ALL_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "GS", "BAC", "V", "MA",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "WMT", "KO", "NKE", "MCD", "SBUX",
    "TSLA", "F", "GM", "RIVN", "LCID", "CRM", "SNOW", "PLTR", "NET", "DDOG"
]
```

- `USER_AGENT`：SEC EDGAR APIの利用規約（ToS）に従い、すべてのリクエストヘッダーに含めます
- `MAX_CONCURRENT_WORKERS = 3`：SECは1秒あたり10リクエストのレートリミットを設けており、安全マージンを持って3並列に制限しています
- 30社は、テクノロジー・金融・ヘルスケア・消費財・自動車・SaaSと分散されており、セクターバランスの取れた分析が可能です

#### セクション6：事前チェック（プリフライト冪等性確認）

```python
today = date.today()
try:
    df_completed = spark.table(LOG_TABLE).filter(
        (col("status") == "SUCCESS") & (col("last_successful_run") == today)
    )
    completed_tasks = set([(row.ticker, row.form_type) for row in df_completed.collect()])
except Exception:
    completed_tasks = set()
```

ダウンロード開始前に、本日すでに成功している（ticker, form_type）のペアをセットとして取得します。`try/except`で囲んでいるのは、初回実行時にログテーブルが空の場合の例外を安全に処理するためです。後のスレッドワーカーでこのセットを参照することで、Sparkセッションのスレッドアンセーフ問題を回避しています。

#### セクション7：スレッドセーフなダウンロードワーカー

```python
def download_filing(ticker, form_type):
    if (ticker, form_type) in completed_tasks:
        return (ticker, form_type, "SKIPPED", 0, "")

    dl = Downloader("FinSage", USER_AGENT, VOLUME_PATH)
    success = False
    retries = 0
    error_msg = ""

    while not success and retries < MAX_RETRIES:
        old_stdout = sys.stdout
        sys.stdout = my_stdout = StringIO()
        try:
            dl.get(form_type, ticker, after="2020-01-01")
            output = my_stdout.getvalue()
            if "Error occurred while downloading" in output or "503" in output:
                raise Exception("SEC API Error/503 Detected.")
            success = True
        except Exception as e:
            retries += 1
            error_msg = str(e)
            time.sleep(10 * retries)
        finally:
            sys.stdout = old_stdout
```

このワーカー関数には複数の巧妙な設計があります：

1. **標準出力のキャプチャ**：`sec-edgar-downloader`ライブラリはエラーを例外ではなく標準出力に出力します。`sys.stdout`を`StringIO`バッファにリダイレクトして出力をキャプチャし、エラーキーワードを検出しています
2. **指数バックオフ**：`time.sleep(10 * retries)` — 1回目失敗で10秒待機、2回目で20秒、3回目で30秒。SEC APIの一時的な負荷を考慮した設計です
3. **`finally`節**：エラーが発生しても必ず標準出力を元に戻します。これがないとノートブック全体の出力が壊れます
4. **冪等性チェック**：関数の最初に`completed_tasks`を確認し、すでに完了済みならスキップ

#### セクション8：並列実行と結果の収集

```python
with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
    futures = [executor.submit(download_filing, t, f) for t in TICKERS for f in FORM_TYPES]
    for future in as_completed(futures):
        results.append(future.result())
```

`ThreadPoolExecutor`で最大3つのスレッドを並列実行します。`as_completed()`はスレッドが完了した順に結果を受け取るためのイテレータで、すべてのスレッドの完了を待ちます。2種類のフォームタイプ（10-K、10-Q）× 30社 = 最大60タスクが3並列で処理されます。

#### セクション9：Delta Lakeへのアトミックな状態更新

```python
if processed_results:
    df_updates = spark.createDataFrame(processed_results, schema=schema)
    df_updates = df_updates.withColumn(
        "last_successful_run",
        expr("IF(status = 'SUCCESS', current_date(), cast(null as date))")
    ).withColumn("updated_at", current_timestamp())

    dt_log = DeltaTable.forName(spark, LOG_TABLE)
    dt_log.alias("t").merge(
        df_updates.alias("s"),
        "t.ticker = s.ticker AND t.form_type = s.form_type"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
```

Delta LakeのMERGE操作（UPSERTとも呼ばれます）を使用して状態を更新します。既存の（ticker, form_type）レコードは更新され、新しいものは挿入されます。`IF(status = 'SUCCESS', current_date(), null)`により、失敗した場合は`last_successful_run`を更新しません。これにより次回実行時に再試行対象として残ります。

---

### ノートブック 02 — `02_bronze_autoloader.py`：ブロンズ層 Auto Loader & SEC API取り込み

このノートブックはブロンズ層の中心です。2つの独立したデータソースからデータを取り込みます：（1）ボリューム上の物理ファイルをDatabricks Auto Loaderでストリーミング取り込み、（2）SEC EDGAR CompanyFacts APIからXBRL JSONデータをバッチ取得します。

#### セクション1：リセットフラグ（緊急用）

```python
RESET_PIPELINE = False
if RESET_PIPELINE:
    spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.finsage_bronze.filings")
    spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.finsage_bronze.ingestion_errors")
    spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.finsage_bronze.xbrl_companyfacts_raw")
    dbutils.fs.rm(f"/Volumes/{CATALOG}/finsage_bronze/checkpoints", recurse=True)
```

`RESET_PIPELINE = True`にすると全テーブルとAuto Loaderのチェックポイントが削除され、ゼロから再取り込みが可能になります。通常はFalseのままにします。このフラグはコードレビューで意図的なリセットを明示するための安全装置です。

#### セクション2：ブロンズテーブルの作成（SQL）

```sql
CREATE TABLE IF NOT EXISTS main.finsage_bronze.filings (
    filing_id         STRING,
    ticker            STRING,
    filing_type       STRING,
    accession_number  STRING,
    fiscal_year       INT,
    file_path         STRING,
    content           BINARY,       -- 生ファイルの全バイナリ内容
    file_size_bytes   LONG,
    ingestion_status  STRING,
    ingested_at       TIMESTAMP
) TBLPROPERTIES (delta.enableChangeDataFeed = true);
```

`content BINARY`カラムがHTMLファイル全体をバイナリとして格納します。シルバー層でテキスト処理するために`decode(col("content"), "UTF-8")`でデコードします。`delta.enableChangeDataFeed = true`により、このテーブルへの変更（INSERT/UPDATE/DELETE）をダウンストリームで追跡できるChange Data Capture（CDC）が有効化されます。

#### セクション3：Auto Loaderによるファイルストリーミング（核心部分）

```python
df_bronze = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", schema_location)
    .option("recursiveFileLookup", "true")
    .load(volume_path)
    .withColumn("file_path",        col("_metadata.file_path"))
    .withColumn("ticker",           split(col("file_path"), "/").getItem(6))
    .withColumn("filing_type",      split(col("file_path"), "/").getItem(7))
    .withColumn("accession_number", split(col("file_path"), "/").getItem(8))
    .withColumn("year_short",       split(col("accession_number"), "-").getItem(1))
    .withColumn("fiscal_year",      concat(lit("20"), col("year_short")).cast("int"))
    .withColumn(
        "filing_id",
        concat_ws("-", col("ticker"), col("filing_type"), col("fiscal_year"), col("accession_number"))
    )
    ...
)
```

**Auto Loader（cloudFiles）の仕組み：**
- `format("cloudFiles")`：Databricksのネイティブなインクリメンタルストリームプロセッサ。ディレクトリの新着ファイルを自動検出します
- `cloudFiles.format = "binaryFile"`：ファイルを生バイナリとして読み込みます。Auto Loaderは`_metadata`という組み込みカラムにファイルパスやサイズなどのメタデータを提供します
- `cloudFiles.schemaLocation`：スキーマ情報をチェックポイントに保存し、スキーマの進化（列の追加など）を自動管理します
- `recursiveFileLookup = true`：サブディレクトリを再帰的に探索します

**ファイルパスからのメタデータ抽出：**

SEC申告書のダウンロード後のファイルパス構造は：
`/Volumes/main/finsage_bronze/raw_filings/sec-edgar-filings/AAPL/10-K/0000320193-21-000105/...`

`split(col("file_path"), "/")`でスラッシュ区切りの配列に分割し、インデックスで各要素を取得：
- インデックス6 → ティッカー（例：`AAPL`）
- インデックス7 → フォームタイプ（例：`10-K`）
- インデックス8 → アクセッション番号（例：`0000320193-21-000105`）

アクセッション番号の`-21-`部分（インデックス1）は申告年度の下2桁。`concat(lit("20"), col("year_short")).cast("int")`で`21` → `"2021"` → `2021`に変換します。

**ストリームの書き込み設定：**

```python
df_bronze.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_path)
    .option("badRecordsPath",     bad_records_path)
    .option("mergeSchema",        "true")
    .trigger(availableNow=True)
    .toTable(target_table)
```

- `outputMode("append")`：ブロンズは追記専用のため、既存データを上書きしません
- `checkpointLocation`：exactly-once処理を保証するためのチェックポイント。同じファイルが2回取り込まれることを防ぎます
- `badRecordsPath`：パースに失敗したレコードをストリームを止めずに別パスに退避します
- `trigger(availableNow=True)`：「現時点で利用可能なすべてのファイルを処理したら停止」というバッチ的な動作。継続的なストリームにはなりません。これにより本番バッチジョブとして安全に使えます

#### セクション4：SEC EDGAR CompanyFacts API取り込み

**Step 0 — 本日分のスキップチェック：**

```python
df_existing = spark.sql("""
    SELECT ticker FROM main.finsage_bronze.xbrl_companyfacts_raw
    WHERE to_date(fetched_at) = current_date() AND api_status = 'success'
""")
already_fetched_today = [row["ticker"] for row in df_existing.collect()]
```

冪等性のため、本日すでに正常取得済みのティッカーはスキップします。毎日一度だけAPIを叩く設計です。

**Step 1 — ティッカー→CIKマッピングの構築：**

```python
company_map_url = "https://www.sec.gov/files/company_tickers.json"
company_map_resp = session.get(company_map_url, headers=HEADERS, timeout=30)
company_map = company_map_resp.json()

for item in company_map.values():
    ticker = item.get("ticker", "").upper()
    if ticker in TICKERS:
        ticker_to_cik[ticker] = str(item.get("cik_str", "")).zfill(10)
```

SECはすべての登録企業のティッカー→CIK（Central Index Key）マッピングを公開しています。CIKはSECのシステム内で企業を一意に識別する数値IDです。`zfill(10)`で10桁のゼロ埋め（例：`320193` → `0000320193`）を行います。これはSECのAPI URLフォーマットの要件です。

**Step 2 — CompanyFacts JSONの取得：**

```python
source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
resp = session.get(source_url, headers=HEADERS, timeout=45)
if resp.status_code == 200:
    payload = resp.json()
    api_rows.append(Row(
        snapshot_id=str(uuid4()), ticker=ticker, cik=cik,
        entity_name=payload.get("entityName"), source_url=source_url,
        raw_json=resp.text, api_status="success",
        ...
    ))
```

`raw_json=resp.text`でJSONペイロード全体を文字列としてブロンズに保存します。シルバー層でPythonの`json.loads()`によって解析されます。`uuid4()`でユニークな`snapshot_id`を生成し、同じティッカーの複数日分のスナップショットを区別します。

**Step 3 & 4 — スキーマ定義とDelta書き込み：**

```python
api_schema = StructType([
    StructField("snapshot_id",      StringType(),  True),
    StructField("raw_json",         StringType(),  True),
    StructField("http_status_code", IntegerType(), True),
    ...
])
spark.createDataFrame(api_rows, schema=api_schema)
    .withColumn("fetched_at", current_timestamp())
    .write.format("delta").mode("append")
    .saveAsTable(f"{CATALOG}.finsage_bronze.xbrl_companyfacts_raw")
```

明示的なSparkスキーマを定義することで型の一貫性を保証します。`.mode("append")`でブロンズの追記専用原則を維持します。`current_timestamp()`でSparkサーバーの現在時刻を`fetched_at`として記録します。

---

### ノートブック 03 — `03_silver_decoder.py`：シルバー層 XBRLデコード & テキスト抽出

このノートブックはブロンズの生データから有用な構造化情報を抽出する「解析エンジン」です。2つの完全に独立したパス（A: XBRL数値データ、B: 10-Kテキストセクション）で処理します。

#### パート A：XBRL CompanyFacts → `financial_statements`テーブル

**TARGET_CONCEPT_MAP — 正規化の中心：**

```python
TARGET_CONCEPT_MAP = {
    "Revenues":                                                    "revenue",
    "SalesRevenueNet":                                             "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax":         "revenue",
    "RevenuesNetOfInterestExpense":                                "revenue",
    "NetIncomeLoss":                                               "net_income",
    "GrossProfit":                                                 "gross_profit",
    "OperatingIncomeLoss":                                         "operating_income",
    "NetCashProvidedByUsedInOperatingActivities":                  "operating_cash_flow",
    "Assets":                                                      "total_assets",
    "StockholdersEquity":                                          "equity",
    "LongTermDebt":                                                "long_term_debt",
    "ResearchAndDevelopmentExpense":                               "rd_expense",
    ...
}
```

これがFinSageの正規化の核心です。SEC XBRLでは同じ概念（例：売上高）が企業によって異なるタグ名で報告されます。このマップは30以上の異なるXBRLコンセプト名を11の統一された指標名に変換します。`tests/unit/test_normalizer.py`でこのマップの網羅性がテストされています。

**flatten_companyfacts関数 — JSONの深いネスト構造を展開：**

```python
def flatten_companyfacts(row):
    out = []
    payload = json.loads(row.raw_json)
    us_gaap = payload.get("facts", {}).get("us-gaap", {})
    for concept, concept_body in us_gaap.items():
        normalized_line_item = TARGET_CONCEPT_MAP.get(concept)
        if not normalized_line_item:
            continue  # TARGET_CONCEPT_MAPにないXBRLコンセプトは無視
        units_map = concept_body.get("units", {})
        for unit, entries in units_map.items():
            for e in entries:
                filing_type = e.get("form")
                fiscal_year = e.get("fy")
                if filing_type not in ("10-K", "10-Q") or fiscal_year is None:
                    continue
                out.append(Row(
                    ticker=row.ticker,
                    normalized_line_item=normalized_line_item,
                    value=float(e.get("val")),
                    fiscal_year=int(fiscal_year),
                    fiscal_period=e.get("fp"),
                    ...
                ))
    return out
```

SEC CompanyFacts APIのJSON構造は深くネストされています：
```
facts → us-gaap → Revenues → units → USD → [{filed: "2021-10-29", fy: 2021, fp: "FY", val: 365817000000, ...}]
```

この関数は`rdd.flatMap()`で各ブロンズ行に対して呼び出され、1行のJSONを多数の財務指標行に展開します。`TARGET_CONCEPT_MAP.get(concept)`でNoneが返ったコンセプトはスキップすることで、関心のある財務指標のみを抽出します。

**SHA-256による決定論的な重複排除：**

```python
.withColumn(
    "statement_id",
    sha2(concat_ws(
        "||",
        coalesce(col("ticker"),      lit("")),
        coalesce(col("accession"),   lit("")),
        coalesce(col("raw_line_item"), lit("")),
        coalesce(col("unit"),        lit("")),
        coalesce(col("period_end"),  lit("")),
    ), 256)
)
```

`statement_id`は（ticker + accession + XBRLコンセプト + 単位 + 期末日）のハッシュです。同じ事実が複数回APIから返ってきても、このIDは同じになります。`||`区切り文字を使うことで、異なるフィールドの組み合わせが同じハッシュを生成する衝突（例：`"A" + "B"` = `"AB"` = `"A" + "B"`）を防いでいます。

**最新スナップショットを優先するウィンドウ関数：**

```python
window_spec = Window.partitionBy("statement_id").orderBy(
    col("source_fetched_at").desc(),
    col("filing_date").desc_nulls_last(),
)
df_financials_latest = (
    df_financials
    .withColumn("rn", row_number().over(window_spec))
    .filter(col("rn") == 1)
    .drop("rn")
)
```

同じ`statement_id`を持つ複数の行（複数日のAPIスナップショットから来る）の中で最新のものだけを保持します。`row_number()`は各グループ内で1から始まる番号を付け、`.filter(col("rn") == 1)`で最上位の1行だけを残します。

**冪等なMERGE書き込み：**

```python
if spark.catalog.tableExists(silver_table):
    DeltaTable.forName(spark, silver_table).alias("t").merge(
        df_financials_latest.alias("s"), "t.statement_id = s.statement_id"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
else:
    df_financials_latest.write.format("delta").saveAsTable(silver_table)
```

テーブルが存在する場合はMERGE（既存レコードを更新、新規レコードを挿入）、初回実行時は直接書き込みます。これによりパイプラインを何度再実行してもデータが重複しません。

#### パート B：10-Kテキストセクション → `filing_sections`テーブル

**SECTION_RULES — 正規表現による境界定義：**

```python
SECTION_RULES = {
    "Business": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1\b(?!\s*[ab]\b)"],
        "end_patterns":   [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1a\b",
                           r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+2\b"],
        "min_words": 250, "fallback_chars": 250000,
    },
    "Risk Factors": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1a\b"],
        ...
        "min_words": 400,
    },
    "MD&A": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+7\b(?!\s*a\b)"],
        ...
    },
}
```

各正規表現の解析：
- `(?im)`：`i`は大文字小文字を区別しない、`m`は`^`が各行の先頭にマッチ
- `[\s>\-\.\(\)\d]{0,12}`：行頭の最大12文字のノイズ（空白、`>`、`-`、数字など）を許容。10-KのHTMLは多様なフォーマットを持つため必要
- `item\s+1\b`：`\s+`で1つ以上の空白、`\b`で単語境界（`item 1a`の`1`にはマッチしない）
- `(?!\s*[ab]\b)`：ネガティブ先読み。「item 1a」や「item 1b」の後ろではマッチしない。Item 1（事業概要）のみを開始点とするため

**HTMLクリーニングパイプライン：**

```python
df_processed = (
    df_bronze_clean
    .withColumn("raw_text",       decode(col("content"), "UTF-8"))
    .withColumn("main_doc",       expr("substring_index(raw_text, '</DOCUMENT>', 1)"))
    .withColumn("no_images",      regexp_replace(col("main_doc"),  r"(?is)<img[^>]*src=[\"']data:image/[^>]*>", " "))
    .withColumn("no_script",      regexp_replace(col("no_images"), r"(?is)<script[^>]*>.*?</script>", " "))
    .withColumn("no_style",       regexp_replace(col("no_script"), r"(?is)<style[^>]*>.*?</style>", " "))
    .withColumn("text_with_breaks", regexp_replace(col("no_style"),
        r"(?i)</?(div|p|br|tr|li|table|...)[^>]*>", "\n"))
    .withColumn("no_html",        regexp_replace(col("text_with_breaks"), "<[^>]+>", " "))
    .withColumn("clean_text",     regexp_replace(col("no_html"),   "\u00a0", " "))
    ...
)
```

クリーニングの段階的なパイプライン：
1. `decode(content, "UTF-8")`：バイナリをテキストに変換
2. `substring_index(raw_text, '</DOCUMENT>', 1)`：SEC EDGARファイルには複数のドキュメントが含まれているため、最初のドキュメント（主文書）だけを抽出
3. `<img...data:image/...>`の削除：Base64エンコードされた埋め込み画像は巨大で、テキスト処理に不要
4. `<script>`と`<style>`タグの削除：JavaScriptとCSSはテキストノイズになる
5. 構造タグ（`<div>`, `<p>`, `<br>`, `<tr>`など）を改行`\n`に変換：段落構造を保持しながらHTMLを除去
6. 残りのすべてのHTMLタグを半角スペースに置換
7. `\u00a0`（ノンブレーキングスペース）を通常スペースに変換

**セクション抽出UDF：**

```python
split_udf = udf(
    extract_sections_hardened,
    StructType([
        StructField("sections", ArrayType(StructType([
            StructField("section_name", StringType()),
            StructField("section_text", StringType()),
            StructField("word_count",   IntegerType()),
        ]))),
        StructField("error", StringType()),
    ])
)
```

Sparkの`udf()`でPython関数をSparkの分散処理に組み込みます。戻り値型は明示的なスキーマで定義されており、`sections`（セクション情報の配列）と`error`（エラーメッセージまたはNull）を返します。

**_choose_best_block — 最良ブロックの選択ロジック：**

```python
def _choose_best_block(text, rule):
    starts = _collect_positions(rule["start_patterns"], text)
    ends   = _collect_positions(rule["end_patterns"],   text)
    if not starts:
        return None
    doc_len, best, best_score = max(len(text), 1), None, -1
    for s in starts:
        end_candidates = [e for e in ends if e > s + 25]
        e = end_candidates[0] if end_candidates else min(len(text), s + rule["fallback_chars"])
        candidate  = text[s:e].strip()
        word_count = len(candidate.split())
        if word_count < rule["min_words"]:
            continue
        score = word_count + ((s / doc_len) * 250)
        if score > best_score:
            best_score = score
            best = {"section_text": candidate, "word_count": word_count, ...}
    return best
```

10-Kのような複雑な文書には、目次と本文の両方に「Item 1」が現れます。目次の「Item 1」を誤って開始点として取り込まないよう、スコアリングシステムを使います：
- `word_count`が`min_words`（250または400語）未満のブロックは除外（目次の1行は単語数が少ない）
- `score = word_count + ((s / doc_len) * 250)`：文書後半に位置するほど（目次は前方にある）スコアが上がるように設計
- 最高スコアのブロックを「真のセクション」として採用

最終的にすべてのセクションをexplodeして1セクション1行に変換し、`overwrite`モードでシルバーテーブルに書き込みます。

---

### ノートブック 04 — `04_gold_metrics.py`：ゴールド層 財務指標集計

このノートブックはシルバーの正規化済み財務指標をさらに精錬し、分析に即座に使える広いテーブルを生成します。厳密な品質フィルタリングと前年比成長率などの派生KPIを計算します。

#### セクション1：時間軸フィルタリング

```python
df = (
    spark.table(silver_table)
    .filter(col("filing_type").rlike("^10-K"))
    .filter(col("fiscal_period") == "FY")
    .filter(col("fiscal_year") >= 2020)
    .withColumn("duration_days", datediff(col("period_end_dt"), col("period_start_dt")))
    ...
)
```

- `filing_type.rlike("^10-K")`：年次申告書のみ。`rlike`は正規表現マッチ（`10-K/A`など修正版も含む）
- `fiscal_period == "FY"`：XBRLの"FY"は完全な会計年度を意味します。Q1〜Q4の四半期データは除外
- `fiscal_year >= 2020`：直近5年分の分析に絞ります
- `duration_days`：`period_start`から`period_end`までの日数。期間が正確に1年間（350〜380日）かどうか検証するために計算します

#### セクション2：コンセプト優先度によるデータ重複排除

```python
concept_priority = (
    when((col("normalized_line_item") == "revenue") &
         (col("raw_line_item") == "RevenueFromContractWithCustomerExcludingAssessedTax"), lit(1))
    .when((col("normalized_line_item") == "revenue") &
          (col("raw_line_item") == "SalesRevenueNet"), lit(2))
    .when((col("normalized_line_item") == "revenue") &
          (col("raw_line_item") == "Revenues"), lit(3))
    ...
    .otherwise(lit(99))
)
```

同じ正規化指標（例：`revenue`）に複数のXBRLコンセプトがマッピングされている場合、どれを優先するかを決定します。優先度1が最優先です。`RevenueFromContractWithCustomerExcludingAssessedTax`（ASC 606準拠の最新基準）を最優先にすることで、会計基準の変更に対応した一貫性のある比較が可能になります。

#### セクション3：期間フィット検証

```python
.withColumn("annual_fit_score", when(
    col("is_duration_metric") &
    col("duration_days").between(350, 380) &
    (col("period_end_year") == col("fiscal_year")),
    lit(1)
).otherwise(lit(0)))
.withColumn("instant_fit_score", when(
    col("is_instant_metric") &
    (col("period_end_year") == col("fiscal_year")),
    lit(1)
).otherwise(lit(0)))
```

2種類の指標を区別します：
- **フロー指標**（`is_duration_metric`）：収益・利益など、期間中の累積値。1年間（350〜380日）の期間でなければなりません。`< 350`は四半期、`> 380`は調整期間の可能性があるため除外
- **瞬間指標**（`is_instant_metric`）：総資産・負債など、特定の時点の残高。会計年度末時点の値であればよい

ウォルマートのような1月決算の企業は、会計年度2023が翌年1月末（2024年）に終わります。そのため`period_end_year == fiscal_year`の厳密チェックのみでなく、`+1`オフセットを許容する設計もあります（コメント参照）。

#### セクション4：正規アクセッション番号の選択（最重要）

```python
df_accession_quality = (
    df.withColumn("usable_fact_flag", when(..., lit(1)).otherwise(lit(0)))
    .groupBy("ticker", "company_name", "fiscal_year", "accession")
    .agg(
        spark_sum(when(required_metric_flag == 1, col("usable_fact_flag")).otherwise(lit(0)))
            .alias("required_metric_hits"),
        countDistinct(when(col("usable_fact_flag") == 1, col("normalized_line_item")))
            .alias("distinct_metric_coverage"),
        spark_max("filing_date_dt").alias("latest_filing_date"),
    )
)

accession_window = Window.partitionBy("ticker", "fiscal_year").orderBy(
    col("required_metric_hits").desc(),
    col("distinct_metric_coverage").desc(),
    col("latest_filing_date").desc(),
)
```

同じ（ticker, fiscal_year）に複数のアクセッション番号（修正申告など）が存在する場合、どれが「最も良い」申告書かを判定します：
1. `required_metric_hits`：売上高・純利益・営業利益など必須6指標のカバレッジ数（最優先）
2. `distinct_metric_coverage`：使用可能な指標の総数
3. `latest_filing_date`：最も新しい申告書（タイブレーカー）

この選択により、集計前に1つの「正規」申告書を確定させ、異なる修正申告のデータが混在することを防ぎます。

#### セクション5：指標の集計（ピボット）

```python
df_base = (
    df_best_fact
    .groupBy("ticker", "company_name", "fiscal_year")
    .agg(
        spark_max(when(col("normalized_line_item") == "revenue",        col("value"))).alias("revenue"),
        spark_max(when(col("normalized_line_item") == "net_income",     col("value"))).alias("net_income"),
        spark_max(when(col("normalized_line_item") == "gross_profit",   col("value"))).alias("gross_profit_raw"),
        spark_max(when(col("normalized_line_item") == "total_assets",   col("value"))).alias("total_assets"),
        spark_max(when(col("normalized_line_item") == "equity",         col("value"))).alias("equity"),
        ...
    )
)
```

`spark_max(when(condition, value))`パターンは**条件付き集計**のイディオムです。`GROUP BY`で1行にまとめながら、指標名をカラム名に変換するピボット操作です。各`normalized_line_item`が異なるカラムになります。

#### セクション6：派生指標の計算

```python
df_metrics = (
    df_base
    .withColumn("gross_profit",
        coalesce(col("gross_profit_raw"), col("revenue") - col("cost_of_revenue")))
    .withColumn("total_debt",
        when(col("short_term_debt").isNull() & col("long_term_debt").isNull(), lit(None).cast("double"))
        .otherwise(coalesce(col("short_term_debt"), lit(0.0)) + coalesce(col("long_term_debt"), lit(0.0))))
    .withColumn("gross_margin_pct",
        when(col("revenue").isNotNull() & (col("revenue") != 0) & col("gross_profit").isNotNull(),
             col("gross_profit") / col("revenue")))
)
```

- `gross_profit`：XBRLで直接報告されている場合はその値を使用、ない場合は`revenue - cost_of_revenue`で計算（`coalesce`でフォールバック）
- `total_debt`：両方がNullの場合は合計もNull（データなし）、それ以外はNullを0として足し合わせ
- `gross_margin_pct`：0除算を防ぐため、`revenue != 0`チェックを含んだ安全な除算

**前年比成長率（YoY）の計算：**

```python
yoy_window = Window.partitionBy("ticker").orderBy("fiscal_year")
df_metrics = (
    df_metrics
    .withColumn("prior_year_revenue", lag("revenue").over(yoy_window))
    .withColumn("revenue_yoy_growth_pct",
        when(col("prior_year_revenue").isNotNull() & (col("prior_year_revenue") != 0),
             (col("revenue") - col("prior_year_revenue")) / col("prior_year_revenue")))
    .withColumn("debt_to_equity",
        when(col("equity").isNotNull() & (col("equity") != 0) & col("total_debt").isNotNull(),
             col("total_debt") / col("equity")))
)
```

`lag("revenue")`は同じティッカーの前年度の売上高を取得するウィンドウ関数です。`partitionBy("ticker")`で企業ごとに分割し、`orderBy("fiscal_year")`で年度順に並べ、1年前（デフォルト`lag`のオフセット=1）の値を参照します。`(revenue - prior_year_revenue) / prior_year_revenue`で成長率を計算します。

#### セクション7：データ品質スコアと最終書き込み

```python
validated_metric_count = (
    when(col("revenue").isNotNull(),             lit(1)).otherwise(lit(0)) +
    when(col("net_income").isNotNull(),          lit(1)).otherwise(lit(0)) +
    ...
    when(col("rd_expense").isNotNull(),          lit(1)).otherwise(lit(0))
)

df_gold = (
    df_metrics
    .withColumn("data_quality_score", validated_metric_count / lit(9.0))
    ...
)
```

9つのコア指標それぞれについて、値が存在する場合に1、Nullの場合に0を加算します。9で割ることで0〜1のスコアを得ます。`data_quality_score = 1.0`は9指標すべてが揃っていることを意味します。このスコアにより、下流の分析でデータ完全性の低いレコードをフィルタリングできます。

---

### ノートブック 05 — `05_vector_chunker.py`：ベクター化チャンク生成 & Vector Searchインデックス

このノートブックはFinSageのRAG（検索拡張生成）基盤を構築します。テキストセクションをLLMが扱えるサイズのチャンクに分割し、Databricks Vector Searchインデックスに登録して、意味的な類似検索を可能にします。

#### セクション1：チャンキング設定

```python
SOURCE_TABLE          = "main.finsage_silver.filing_sections"
TARGET_TABLE          = "main.finsage_gold.filing_section_chunks"
EMBEDDING_MODEL       = "text-embedding-3-large"
CHUNK_TOKENS          = 512
CHUNK_OVERLAP_TOKENS  = 64
CHUNK_VERSION         = f"tok_{CHUNK_TOKENS}_{CHUNK_OVERLAP_TOKENS}_v1"
```

- `CHUNK_TOKENS = 512`：1チャンクのトークン数上限。OpenAIの`text-embedding-3-large`モデルは最大8,192トークンを処理できますが、512トークンが検索精度と文脈保持のバランスの良いサイズです
- `CHUNK_OVERLAP_TOKENS = 64`：連続するチャンク間のオーバーラップ。チャンク境界でセンテンスが切れることによる意味の損失を防ぎます
- `CHUNK_VERSION`：チャンキング設定のバージョン文字列。パラメータを変更した際に既存チャンクとの区別に使用

#### セクション2：tiktokenエンコーダーの遅延初期化

```python
_ENCODING = None

def get_encoding():
    global _ENCODING
    if _ENCODING is None:
        try:
            _ENCODING = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        except KeyError:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING
```

`tiktoken`はOpenAIが開発したトークナイザーです。テキストをトークン（モデルの語彙単位）に変換します。遅延初期化（初回呼び出し時のみ初期化）にすることで、Sparkの分散実行環境でのシリアライゼーション問題を回避します。`cl100k_base`はGPT-4で使用されるエンコーディングで、フォールバックとして使用します。

#### セクション3：決定論的チャンクID生成

```python
def deterministic_chunk_id(
    filing_id: str,
    section_name: str,
    chunk_index: int,
    chunk_text: str,
    chunk_version: str,
) -> str:
    payload = {
        "filing_id":    str(filing_id),
        "section_name": str(section_name),
        "chunk_index":  int(chunk_index),
        "chunk_text":   chunk_text,
        "chunk_version": chunk_version,
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

この関数はチャンクの内容と位置情報から決定論的なIDを生成します：
- `json.dumps(sort_keys=True)`：キーの順序を固定して、同じ内容なら常に同じJSON文字列になるようにします
- `separators=(",", ":")`：余分な空白を除去してコンパクトなJSON（バイト単位での一意性を保証）
- SHA-256ハッシュ：同じ入力からは常に同じID。これによりパイプラインを再実行してもチャンクのIDが変わらず、Vector SearchインデックスのMERGEが正確に機能します

#### セクション4：Pandas UDF によるトークンベースチャンキング

```python
@F.pandas_udf(chunk_array_schema)
def chunk_sections_udf(
    section_text_col: pd.Series,
    filing_id_col: pd.Series,
    section_name_col: pd.Series,
) -> pd.Series:
    enc  = get_encoding()
    step = CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS  # = 512 - 64 = 448

    for text, filing_id, section_name in zip(section_text_col, filing_id_col, section_name_col):
        normalized = normalize_text(str(text))
        token_ids  = enc.encode(normalized)

        row_chunks  = []
        chunk_index = 0
        for start in range(0, len(token_ids), step):
            end        = min(start + CHUNK_TOKENS, len(token_ids))
            chunk_ids  = token_ids[start:end]
            chunk_text = enc.decode(chunk_ids).strip()
            cid = deterministic_chunk_id(...)
            row_chunks.append({
                "chunk_id":    cid,
                "chunk_index": chunk_index,
                "chunk_text":  chunk_text,
                "token_count": len(chunk_ids),
                ...
            })
            chunk_index += 1
            if end == len(token_ids):
                break
        out.append(row_chunks)
```

`@F.pandas_udf`は通常のSparkのUDFより高速なベクトル化UDFです。通常のUDFは1行ずつ処理しますが、Pandas UDFはデータを`pd.Series`のバッチで受け取り、パフォーマンスが大幅に向上します。

チャンキングアルゴリズム：
- `step = 448`：各チャンクの開始点を448トークンずつ進める（512 - 64オーバーラップ）
- テキスト全体をtiktokenでトークンIDのリストに変換
- スライス`token_ids[start:end]`でトークンIDを取り出し
- `enc.decode(chunk_ids)`でトークンIDを再びテキストに戻す
- 最後のチャンク（`end == len(token_ids)`）で`break`

視覚的なチャンク構造（512トークン、64オーバーラップ）：
```
[チャンク0: トークン   0〜511]
[チャンク1: トークン 448〜959]
[チャンク2: トークン 896〜1407]
              ^^^--- 64トークンのオーバーラップ
```

#### セクション5：データ品質ガード

```python
dup_count = (
    df_chunks.groupBy("chunk_id").count()
    .filter(F.col("count") > 1).limit(1).count()
)
if dup_count > 0:
    raise RuntimeError("Duplicate chunk_id detected. Aborting write.")

bad_rows = (
    df_chunks.filter(
        F.col("chunk_text").isNull() |
        (F.col("token_count") <= 0) |
        (F.col("chunk_index") < 0)
    ).limit(1).count()
)
if bad_rows > 0:
    raise RuntimeError("Invalid chunk rows detected. Aborting write.")
```

書き込み前に2つのアサーションを実行します：
1. `chunk_id`の重複チェック：決定論的IDのロジックが正しく機能しているか確認。重複があれば Vector SearchのMERGEが誤動作する
2. 不正行チェック：`chunk_text`がNull、`token_count`が0以下、`chunk_index`が負数のいずれかが存在すれば異常データとして書き込みを中断

`.limit(1).count()`の使用が重要です。`count()`は全件カウントですが、`.limit(1).count()`は1件でも存在すれば1を返すので、大量データに対しても非常に高速です。

#### セクション6：Vector Searchエンドポイントとインデックスのプロビジョニング

```python
VECTOR_SEARCH_ENDPOINT_NAME  = "finsage_vs_endpoint"
INDEX_NAME                   = "main.finsage_gold.filing_chunks_index"
EMBEDDING_MODEL_ENDPOINT     = "databricks-bge-large-en"
PIPELINE_TYPE                = "TRIGGERED"
```

- `finsage_vs_endpoint`：Databricks Vector Searchのエンドポイント（クラスターに類似したコンピュートリソース）
- `databricks-bge-large-en`：BGE（BAAI General Embedding）Large英語モデル。Databricksがホストするモデルサービングエンドポイントでテキストをベクターに変換します
- `TRIGGERED`パイプライン：インデックスの更新は明示的にトリガーされるまで自動では更新されません（コスト効率的）

**リトライロジック（指数バックオフ + ジッター）：**

```python
def _retryable_call(fn, retries: int = 8, base_sleep: float = 1.5, max_sleep: float = 20.0):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == retries:
                break
            backoff = min(max_sleep, base_sleep * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            time.sleep(backoff)
    raise last_exc
```

`2 ** (attempt - 1)`で指数バックオフ（1.5s, 3s, 6s, 12s, 20s...）を実現します。`random.uniform(0, 0.5)`のジッター（ランダム変動）を加えることで、複数のクライアントが同時にリトライする「サンダリングハード問題」を防ぎます。

**エンドポイントのONLINE待機ポーリング：**

```python
def wait_for_endpoint_online(vsc, endpoint_name, timeout_sec):
    start = time.time()
    while True:
        if time.time() - start > timeout_sec:
            raise TimeoutError(...)
        ep    = _retryable_call(lambda: vsc.get_endpoint(name=endpoint_name))
        state = _normalize_state(_nested_get(ep, ("endpoint_status", "state"), ("status", "state")))
        if state == "ONLINE":
            return
        if state in {"FAILED", "ERROR"}:
            raise RuntimeError(...)
        time.sleep(POLL_SEC)  # 15秒ごとにポーリング
```

Vector Searchエンドポイントの起動には数分かかります。15秒ごとにステータスをポーリングし、ONLINE状態になるまで最大30分待ちます。`_nested_get`でAPIレスポンスの異なる構造（SDKバージョンの違い）を吸収します。

**類似検索デモ：**

```python
def search_financial_filings(query: str, num_results: int = 3):
    index   = vsc.get_index(endpoint_name="finsage_vs_endpoint", index_name=INDEX_NAME)
    results = index.similarity_search(
        query_text=query,
        columns_to_return=["ticker", "fiscal_year", "section_name", "chunk_text"],
        num_results=num_results,
    )
    docs = results.get("result", {}).get("data_array", [])
    return "\n---\n".join(
        f"[{d[0]} | {d[1]} | {d[2]}]\n{d[3]}" for d in docs
    )

print(search_financial_filings("What did Apple say about supply chain or manufacturing risks?"))
```

インデックスが構築されると、自然言語クエリでサプライチェーンリスクについてAppleが言及した部分を検索できます。`similarity_search`はクエリテキストを同じ埋め込みモデル（BGE Large）でベクター化し、コサイン類似度で最近傍のチャンクを返します。これがRAGシステムの検索（Retrieval）コンポーネントです。

---

## ライセンス

社内プロジェクト — Arsaga Partners。全権利留保。
