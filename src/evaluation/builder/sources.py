"""Independent source adapters for SEC CompanyConcept and yfinance."""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

from .concepts import METRIC_SPECS
from .tickers import TickerRecord

SEC_BASE_URL = "https://data.sec.gov/api/xbrl"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
SEC_USER_AGENT = "FinSage Eval Builder digvijay@arsaga.jp"
SEC_MIN_SECONDS_PER_REQUEST = 0.11
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class SecFact:
    """Normalized SEC CompanyConcept fact."""

    value: float
    fy: Optional[int]
    fp: Optional[str]
    form: Optional[str]
    accn: Optional[str]
    filed: Optional[str]
    end: Optional[str]
    start: Optional[str]
    frame: Optional[str]
    unit: str
    concept: str
    taxonomy: str


@dataclass(frozen=True)
class ChosenValue:
    """Resolved value with provenance."""

    value: float
    concept: str
    form: str
    accn: str
    filed: str
    source_secondary: Optional[str]
    secondary_value: Optional[float]
    agreement: Optional[bool]
    diff_pct: Optional[float]


class SecClient:
    """SEC API client with deterministic request pacing and retry."""

    def __init__(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": SEC_USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            }
        )
        self._last_request_ts = 0.0
        self._concept_cache: Dict[str, List[SecFact]] = {}
        self._submissions_cache: Dict[str, dict] = {}

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < SEC_MIN_SECONDS_PER_REQUEST:
            time.sleep(SEC_MIN_SECONDS_PER_REQUEST - elapsed)

    def _request_json(self, url: str) -> Optional[dict]:
        attempts = 3
        for attempt in range(1, attempts + 1):
            self._throttle()
            response = self._session.get(url, timeout=self._timeout_seconds)
            self._last_request_ts = time.monotonic()
            if response.status_code == 404:
                return None
            if response.status_code == 200:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504, 403} and attempt < attempts:
                time.sleep(0.5 * attempt)
                continue
            response.raise_for_status()
        return None

    def fetch_company_concept(
        self, cik: str, taxonomy: str, concept: str, preferred_unit: Optional[str]
    ) -> List[SecFact]:
        """Fetch all facts for one (CIK, taxonomy, concept)."""
        cache_key = f"{cik}:{taxonomy}:{concept}:{preferred_unit or 'any'}"
        if cache_key in self._concept_cache:
            return self._concept_cache[cache_key]
        url = f"{SEC_BASE_URL}/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"
        payload = self._request_json(url)
        if payload is None:
            self._concept_cache[cache_key] = []
            return []
        units = payload.get("units", {})
        selected_units: Iterable[str]
        if preferred_unit and preferred_unit in units:
            selected_units = [preferred_unit]
        elif preferred_unit:
            selected_units = sorted(units.keys())
        else:
            selected_units = sorted(units.keys())

        facts: List[SecFact] = []
        for unit in selected_units:
            for item in units.get(unit, []):
                val = item.get("val")
                if val is None:
                    continue
                try:
                    value = float(val)
                except (TypeError, ValueError):
                    continue
                facts.append(
                    SecFact(
                        value=value,
                        fy=int(item["fy"]) if item.get("fy") is not None else None,
                        fp=item.get("fp"),
                        form=item.get("form"),
                        accn=item.get("accn"),
                        filed=item.get("filed"),
                        end=item.get("end"),
                        start=item.get("start"),
                        frame=item.get("frame"),
                        unit=unit,
                        concept=concept,
                        taxonomy=taxonomy,
                    )
                )
        self._concept_cache[cache_key] = facts
        return facts

    def fetch_submissions(self, cik: str) -> Optional[dict]:
        """Fetch SEC submissions payload used for 10-K filing date questions."""
        if cik in self._submissions_cache:
            return self._submissions_cache[cik]
        url = f"{SEC_SUBMISSIONS_URL}/CIK{cik}.json"
        payload = self._request_json(url)
        if payload is not None:
            self._submissions_cache[cik] = payload
        return payload


def _safe_date(date_text: Optional[str]) -> Optional[dt.date]:
    if not date_text:
        return None
    return dt.datetime.strptime(date_text, "%Y-%m-%d").date()


def _span_days(fact: SecFact) -> Optional[int]:
    if not fact.start or not fact.end:
        return None
    return (_safe_date(fact.end) - _safe_date(fact.start)).days  # type: ignore[operator]


def _preferred_unit(metric: str) -> Optional[str]:
    if metric in {"employees"}:
        return "pure"
    if metric in {"shares_outstanding"}:
        return "shares"
    if METRIC_SPECS[metric].taxonomy == "us-gaap":
        return "USD"
    return None


def _form_matches(form: Optional[str], target_form: str) -> bool:
    return form in {target_form, f"{target_form}/A"}


def _fact_fits_period(fact: SecFact, metric: str, form: str, fy: int, quarter: Optional[int]) -> bool:
    if fact.fy != fy:
        return False
    if not _form_matches(fact.form, form):
        return False
    is_flow = METRIC_SPECS[metric].is_flow
    if form == "10-K":
        if quarter is not None:
            return False
        if not is_flow:
            return True
        days = _span_days(fact)
        return days is not None and 340 <= days <= 380
    if form == "10-Q":
        if quarter is None:
            return False
        if fact.fp != f"Q{quarter}":
            return False
        if not is_flow:
            return True
        days = _span_days(fact)
        return days is not None and 80 <= days <= 100
    return False


def _most_recent_filed(facts: List[SecFact]) -> Optional[SecFact]:
    if not facts:
        return None
    ordered = sorted(
        facts,
        key=lambda item: (
            item.filed or "",
            item.end or "",
            item.accn or "",
        ),
        reverse=True,
    )
    return ordered[0]


class YFinanceClient:
    """yfinance adapter with cached statement frames and fiscal mapping."""

    def __init__(self) -> None:
        self._ticker_cache: Dict[str, yf.Ticker] = {}

    def _get_ticker(self, ticker: str) -> yf.Ticker:
        if ticker not in self._ticker_cache:
            self._ticker_cache[ticker] = yf.Ticker(ticker)
        return self._ticker_cache[ticker]

    def _metric_row_candidates(self, metric: str) -> List[str]:
        rows: Dict[str, List[str]] = {
            "revenue": ["Total Revenue", "Revenue", "Revenues", "Sales Revenue"],
            "net_income": ["Net Income", "Net Income Common Stockholders"],
            "operating_income": ["Operating Income", "Operating Income Loss"],
            "gross_profit": ["Gross Profit"],
            "rd_expense": ["Research And Development", "Research Development"],
            "total_assets": ["Total Assets"],
            "total_liabilities": [
                "Total Liabilities Net Minority Interest",
                "Total Liabilities",
            ],
            "total_equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
            "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
            "total_debt": ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
            "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
            "employees": ["Full Time Employees"],
        }
        return rows.get(metric, [])

    def _fiscal_year_and_quarter(self, period_end: pd.Timestamp, fy_end_month: int) -> Tuple[int, int]:
        month = int(period_end.month)
        year = int(period_end.year)
        fiscal_year = year if month <= fy_end_month else year + 1
        fiscal_year_start_month = 1 if fy_end_month == 12 else fy_end_month + 1
        month_offset = (month - fiscal_year_start_month) % 12
        quarter = month_offset // 3 + 1
        return fiscal_year, int(quarter)

    def _extract_value(
        self,
        frame: pd.DataFrame,
        metric: str,
        fy: int,
        quarter: Optional[int],
        fy_end_month: int,
    ) -> Optional[float]:
        if frame.empty:
            return None
        row_name = None
        for candidate in self._metric_row_candidates(metric):
            if candidate in frame.index:
                row_name = candidate
                break
        if row_name is None:
            return None

        best_value: Optional[float] = None
        best_date: Optional[pd.Timestamp] = None
        row = frame.loc[row_name]
        for col in sorted(frame.columns):
            val = row[col]
            if pd.isna(val):
                continue
            fiscal_year, fiscal_quarter = self._fiscal_year_and_quarter(pd.Timestamp(col), fy_end_month)
            if fiscal_year != fy:
                continue
            if quarter is None and fiscal_quarter != 4:
                continue
            if quarter is not None and fiscal_quarter != quarter:
                continue
            if best_date is None or col > best_date:
                best_date = col
                best_value = float(val)
        return best_value

    def get_value(self, ticker_meta: TickerRecord, metric: str, fy: int, quarter: Optional[int]) -> Optional[float]:
        ticker = self._get_ticker(ticker_meta.ticker)
        if metric in {"revenue", "net_income", "operating_income", "gross_profit", "rd_expense"}:
            frame = ticker.quarterly_income_stmt if quarter is not None else ticker.income_stmt
        elif metric in {"total_assets", "total_liabilities", "total_equity", "total_debt", "shares_outstanding"}:
            frame = ticker.quarterly_balance_sheet if quarter is not None else ticker.balance_sheet
        elif metric in {"operating_cash_flow"}:
            frame = ticker.quarterly_cash_flow if quarter is not None else ticker.cash_flow
        else:
            return None
        return self._extract_value(frame, metric, fy, quarter, ticker_meta.fiscal_year_end_month)


def select_sec_fact(
    sec_client: SecClient,
    ticker_meta: TickerRecord,
    metric: str,
    fy: int,
    form: str,
    quarter: Optional[int] = None,
) -> Optional[SecFact]:
    """Resolve one metric fact by trying concepts in order and choosing latest filed."""
    from .concepts import concept_candidates_for_metric  # local import to avoid circularity

    preferred_unit = _preferred_unit(metric)
    for concept in concept_candidates_for_metric(metric, ticker_meta.ticker):
        facts = sec_client.fetch_company_concept(
            cik=ticker_meta.cik,
            taxonomy=METRIC_SPECS[metric].taxonomy,
            concept=concept,
            preferred_unit=preferred_unit,
        )
        filtered = [f for f in facts if _fact_fits_period(f, metric, form=form, fy=fy, quarter=quarter)]
        chosen = _most_recent_filed(filtered)
        if chosen is not None:
            return chosen
    return None


def cross_validated_value(
    sec_fact: Optional[SecFact],
    yfinance_value: Optional[float],
    metric: str,
) -> Optional[ChosenValue]:
    """Apply tolerance-based agreement and SEC precedence policy."""
    if sec_fact is None:
        return None
    sec_value = sec_fact.value
    if yfinance_value is None:
        return ChosenValue(
            value=sec_value,
            concept=sec_fact.concept,
            form=sec_fact.form or "",
            accn=sec_fact.accn or "",
            filed=sec_fact.filed or "",
            source_secondary=None,
            secondary_value=None,
            agreement=None,
            diff_pct=None,
        )

    if sec_value == 0:
        diff_pct = 0.0 if yfinance_value == 0 else 100.0
    else:
        diff_pct = abs(sec_value - yfinance_value) / abs(sec_value) * 100.0
    tolerance = 0.5 if metric in {"revenue", "net_income", "total_assets", "total_liabilities", "total_equity"} else 1.0
    agreement = diff_pct <= tolerance
    return ChosenValue(
        value=sec_value,
        concept=sec_fact.concept,
        form=sec_fact.form or "",
        accn=sec_fact.accn or "",
        filed=sec_fact.filed or "",
        source_secondary="yfinance",
        secondary_value=float(yfinance_value),
        agreement=agreement,
        diff_pct=diff_pct,
    )


def get_10k_filing_date(sec_client: SecClient, cik: str, fy: int) -> Optional[Tuple[str, str]]:
    """Return (filing_date, accession_number) for the specified fiscal year 10-K."""
    payload = sec_client.fetch_submissions(cik)
    if payload is None:
        return None
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    report_dates = recent.get("reportDate", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])

    candidates: List[Tuple[str, str]] = []
    for form, report_date, filing_date, accession in zip(forms, report_dates, filing_dates, accession_numbers):
        if form not in {"10-K", "10-K/A"}:
            continue
        if not report_date:
            continue
        report_year = int(str(report_date)[:4])
        if report_year != fy:
            continue
        candidates.append((filing_date, accession))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]

