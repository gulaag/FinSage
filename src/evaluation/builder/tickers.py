"""Canonical ticker registry for deterministic evaluation generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TickerRecord:
    """Metadata required for source selection and fiscal mapping."""

    ticker: str
    company_name: str
    cik: str
    sector: str
    fiscal_year_end_month: int


TICKER_REGISTRY: Dict[str, TickerRecord] = {
    "AAPL": TickerRecord("AAPL", "Apple Inc.", "0000320193", "tech_megacap", 9),
    "ABBV": TickerRecord("ABBV", "AbbVie Inc.", "0001551152", "pharma", 12),
    "AMZN": TickerRecord("AMZN", "Amazon.com Inc.", "0001018724", "tech_megacap", 12),
    "BAC": TickerRecord("BAC", "Bank of America Corporation", "0000070858", "bank", 12),
    "CRM": TickerRecord("CRM", "Salesforce Inc.", "0001108524", "saas", 1),
    "DDOG": TickerRecord("DDOG", "Datadog Inc.", "0001561550", "saas", 12),
    "F": TickerRecord("F", "Ford Motor Company", "0000037996", "auto", 12),
    "GM": TickerRecord("GM", "General Motors Company", "0001467858", "auto", 12),
    "GOOGL": TickerRecord("GOOGL", "Alphabet Inc.", "0001652044", "tech_megacap", 12),
    "GS": TickerRecord("GS", "Goldman Sachs Group Inc.", "0000886982", "bank", 12),
    "JNJ": TickerRecord("JNJ", "Johnson & Johnson", "0000200406", "pharma", 12),
    "JPM": TickerRecord("JPM", "JPMorgan Chase & Co.", "0000019617", "bank", 12),
    "KO": TickerRecord("KO", "The Coca-Cola Company", "0000021344", "retail", 12),
    "LCID": TickerRecord("LCID", "Lucid Group Inc.", "0001811210", "auto", 12),
    "MA": TickerRecord("MA", "Mastercard Incorporated", "0001141391", "payments", 12),
    "MCD": TickerRecord("MCD", "McDonald's Corporation", "0000063908", "retail", 12),
    "MRK": TickerRecord("MRK", "Merck & Co. Inc.", "0000310158", "pharma", 12),
    "MSFT": TickerRecord("MSFT", "Microsoft Corporation", "0000789019", "tech_megacap", 6),
    "NET": TickerRecord("NET", "Cloudflare Inc.", "0001477333", "saas", 12),
    "NKE": TickerRecord("NKE", "NIKE Inc.", "0000320187", "retail", 5),
    "NVDA": TickerRecord("NVDA", "NVIDIA Corporation", "0001045810", "tech_megacap", 1),
    "PFE": TickerRecord("PFE", "Pfizer Inc.", "0000078003", "pharma", 12),
    "PLTR": TickerRecord("PLTR", "Palantir Technologies Inc.", "0001321655", "saas", 12),
    "RIVN": TickerRecord("RIVN", "Rivian Automotive Inc.", "0001874178", "auto", 12),
    "SBUX": TickerRecord("SBUX", "Starbucks Corporation", "0000829224", "retail", 9),
    "SNOW": TickerRecord("SNOW", "Snowflake Inc.", "0001640147", "saas", 1),
    "TSLA": TickerRecord("TSLA", "Tesla Inc.", "0001318605", "auto", 12),
    "UNH": TickerRecord("UNH", "UnitedHealth Group Inc.", "0000731766", "insurance", 12),
    "V": TickerRecord("V", "Visa Inc.", "0001403161", "payments", 9),
    "WMT": TickerRecord("WMT", "Walmart Inc.", "0000104169", "retail", 1),
}


ORDERED_TICKERS: List[str] = sorted(TICKER_REGISTRY.keys())

