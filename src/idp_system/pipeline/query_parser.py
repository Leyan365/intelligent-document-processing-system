from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import date as CalendarDate, datetime
import re

@dataclass
class ParsedSearchQuery:
    raw_query: str
    semantic_text: str
    document_type: str | None = None
    supplier: str | None = None
    document_number: str | None = None
    amount_eq: Decimal | None = None
    amount_lt: Decimal | None = None
    amount_lte: Decimal | None = None
    amount_gt: Decimal | None = None
    amount_gte: Decimal | None = None
    amount_min: Decimal | None = None
    amount_max: Decimal | None = None
    date_eq: CalendarDate | None = None
    date_lt: CalendarDate | None = None
    date_lte: CalendarDate | None = None
    date_gt: CalendarDate | None = None
    date_gte: CalendarDate | None = None
    date_min: CalendarDate | None = None
    date_max: CalendarDate | None = None
    date_error: str | None = None


MONTH_NAME_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
DATE_VALUE_PATTERN = (
    rf"(?:\d{{4}}[./-]\d{{1,2}}[./-]\d{{1,2}}|"
    rf"\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?(?:\s+of)?\s+{MONTH_NAME_PATTERN}\s*,?\s+\d{{4}}|"
    rf"{MONTH_NAME_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+\d{{4}})"
)
INCOMPLETE_DATE_VALUE_PATTERN = (
    rf"(?:\d{{4}}|{MONTH_NAME_PATTERN}\s+\d{{4}}|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?(?:\s+of)?\s+{MONTH_NAME_PATTERN})"
)

def parse_amount(val_str: str) -> Decimal | None:
    cleaned = re.sub(r'[^\d,\.]', '', val_str)
    cleaned = cleaned.replace(',', '')
    # Find the contiguous number
    match = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    if not match:
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def parse_date(value: str) -> CalendarDate | None:
    """Parse supported document and natural-language dates as calendar dates.

    Ambiguous numeric input is interpreted day-first (DD/MM/YYYY), matching the
    document formats used by this application.
    """
    cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", value.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\bof\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"\bSept\b", "Sep", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    formats = (
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
        "%d-%b-%Y", "%d-%B-%Y",
        "%d %b %Y", "%d %B %Y",
        "%b %d %Y", "%B %d %Y",
    )
    for date_format in formats:
        try:
            parsed = datetime.strptime(cleaned, date_format).date()
        except ValueError:
            continue
        if 1990 <= parsed.year <= 2100:
            return parsed
    return None

import string

def normalize_supplier_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    for p in string.punctuation:
        name = name.replace(p, ' ')
    return re.sub(r'\s+', ' ', name).strip()

def parse_query(query: str, known_suppliers: list[str] | None = None) -> ParsedSearchQuery:
    semantic_text = query
    parsed = ParsedSearchQuery(raw_query=query, semantic_text="")

    # 1. Document type
    doc_types = {
        r'\b(?:purchase\s*orders?|pos?)\b': 'purchase_order',
        r'\binvoices?\b': 'invoice',
        r'\breceipts?\b': 'receipt'
    }
    for pattern, d_type in doc_types.items():
        if re.search(pattern, semantic_text, re.IGNORECASE):
            parsed.document_type = d_type
            semantic_text = re.sub(pattern, '', semantic_text, flags=re.IGNORECASE)
            break

    # Capture identifiers from the raw query because document-type removal
    # above intentionally strips prefixes such as "invoice" and "PO".
    doc_num_match = re.search(
        r'\b(?:po|invoice|receipt)\s+(?:number\s+)?([A-Za-z]*\d[A-Za-z0-9\-_]*)\b',
        query,
        re.IGNORECASE,
    )
    if doc_num_match:
        parsed.document_number = doc_num_match.group(1)
        identifier_pattern = re.compile(
            rf'\b(?:number\s+)?{re.escape(parsed.document_number)}\b',
            re.IGNORECASE,
        )
        semantic_text = identifier_pattern.sub('', semantic_text, count=1)

    # 2. Dates. Date filters are evaluated before numeric amounts so that a
    # range such as "between 1 January 2026 and 31 January 2026" cannot be
    # mistaken for an amount range.
    date_range_match = re.search(
        rf'\b(?:between|from)\s+({DATE_VALUE_PATTERN})\s+(?:and|to)\s+({DATE_VALUE_PATTERN})\b',
        semantic_text,
        re.IGNORECASE,
    )
    if date_range_match:
        start_date = parse_date(date_range_match.group(1))
        end_date = parse_date(date_range_match.group(2))
        if start_date is not None and end_date is not None and start_date <= end_date:
            parsed.date_min = start_date
            parsed.date_max = end_date
            semantic_text = semantic_text[:date_range_match.start()] + semantic_text[date_range_match.end():]
    else:
        date_patterns = (
            ("date_gte", rf'\b(?:on\s+or\s+after|on/after|from)\s+({DATE_VALUE_PATTERN})\b'),
            ("date_gt", rf'\b(?:after|later\s+than|above|over)\s+({DATE_VALUE_PATTERN})\b'),
            ("date_lte", rf'\b(?:on\s+or\s+before|on/before|up\s+to|until|through)\s+({DATE_VALUE_PATTERN})\b'),
            ("date_lt", rf'\b(?:before|earlier\s+than|prior\s+to|below|under)\s+({DATE_VALUE_PATTERN})\b'),
            ("date_eq", rf'\b(?:on|dated|date(?:\s+is)?)\s+({DATE_VALUE_PATTERN})\b'),
        )
        for attribute, pattern in date_patterns:
            date_match = re.search(pattern, semantic_text, re.IGNORECASE)
            if not date_match:
                continue
            parsed_date = parse_date(date_match.group(1))
            if parsed_date is not None:
                setattr(parsed, attribute, parsed_date)
                semantic_text = semantic_text[:date_match.start()] + semantic_text[date_match.end():]
            break

        if not any(getattr(parsed, attribute) is not None for attribute, _ in date_patterns):
            incomplete_date_match = re.search(
                # "above 3000" and "below 4000" are amounts. The shared
                # above/below wording is considered a partial date only when
                # it includes a month name.
                rf'\b(?:after|later\s+than|before|earlier\s+than|prior\s+to|'
                rf'on\s+or\s+after|on\s+or\s+before|on|dated)\s+{INCOMPLETE_DATE_VALUE_PATTERN}\b|'
                rf'\b(?:above|over|below|under)\s+'
                rf'(?:{MONTH_NAME_PATTERN}\s+\d{{4}}|\d{{1,2}}(?:st|nd|rd|th)?(?:\s+of)?\s+{MONTH_NAME_PATTERN})\b',
                semantic_text,
                re.IGNORECASE,
            )
            if incomplete_date_match:
                parsed.date_error = (
                    "Date filters require a complete day, month, and year, "
                    "for example: after 1st January 2026."
                )

    # 3. Amounts
    # We look for specific patterns
    # Ranges: between X and Y, from X to Y
    range_match = re.search(r'\b(?:between|from)\s+([A-Za-z\$€£¥₹\.]*\s*[\d\,\.]+)\s+(?:and|to)\s+([A-Za-z\$€£¥₹\.]*\s*[\d\,\.]+)\b', semantic_text, re.IGNORECASE)
    if range_match:
        min_amt = parse_amount(range_match.group(1))
        max_amt = parse_amount(range_match.group(2))
        if min_amt is not None and max_amt is not None:
            parsed.amount_min = min_amt
            parsed.amount_max = max_amt
            semantic_text = semantic_text[:range_match.start()] + semantic_text[range_match.end():]

    else:
        # Single amounts
        lt_match = re.search(r'\b(?:below|under|less than|amount below|amount under)\s+([A-Za-z\$€£¥₹\.]*\s*[\d\,\.]+)\b', semantic_text, re.IGNORECASE)
        if lt_match:
            amt = parse_amount(lt_match.group(1))
            if amt is not None:
                parsed.amount_lt = amt
                semantic_text = semantic_text[:lt_match.start()] + semantic_text[lt_match.end():]
        else:
            lte_match = re.search(r'\b(?:up to|at most)\s+([A-Za-z\$€£¥₹\.]*\s*[\d\,\.]+)\b|\b([A-Za-z\$€£¥₹\.]*\s*[\d\,\.]+)\s+or less\b', semantic_text, re.IGNORECASE)
            if lte_match:
                amt_str = lte_match.group(1) or lte_match.group(2)
                amt = parse_amount(amt_str)
                if amt is not None:
                    parsed.amount_lte = amt
                    semantic_text = semantic_text[:lte_match.start()] + semantic_text[lte_match.end():]
            else:
                gt_match = re.search(r'\b(?:above|over|greater than|amount above)\s+([A-Za-z\$€£¥₹\.]*\s*[\d\,\.]+)\b', semantic_text, re.IGNORECASE)
                if gt_match:
                    amt = parse_amount(gt_match.group(1))
                    if amt is not None:
                        parsed.amount_gt = amt
                        semantic_text = semantic_text[:gt_match.start()] + semantic_text[gt_match.end():]
                else:
                    gte_match = re.search(r'\b(?:at least)\s+([A-Za-z\$€£¥₹\.]*\s*[\d\,\.]+)\b|\b([A-Za-z\$€£¥₹\.]*\s*[\d\,\.]+)\s+or more\b', semantic_text, re.IGNORECASE)
                    if gte_match:
                        amt_str = gte_match.group(1) or gte_match.group(2)
                        amt = parse_amount(amt_str)
                        if amt is not None:
                            parsed.amount_gte = amt
                            semantic_text = semantic_text[:gte_match.start()] + semantic_text[gte_match.end():]
                    else:
                        eq_match = re.search(r'\b(?:amount|for)\s*(?:=|is)?\s*([A-Za-z\$€£¥₹\.]*\s*[\d\,\.]+)\b', semantic_text, re.IGNORECASE)
                        if eq_match:
                            amt = parse_amount(eq_match.group(1))
                            if amt is not None:
                                parsed.amount_eq = amt
                                semantic_text = semantic_text[:eq_match.start()] + semantic_text[eq_match.end():]

    # Supplier exact phrase extraction
    if known_suppliers:
        normalized_map = {}
        for s in known_suppliers:
            norm_s = normalize_supplier_name(s)
            if norm_s:
                # If multiple have the same norm, the last one overwrites, which is fine or we can keep the first
                if norm_s not in normalized_map:
                    normalized_map[norm_s] = s

        sorted_norms = sorted(normalized_map.keys(), key=len, reverse=True)
        norm_text = normalize_supplier_name(semantic_text)

        for norm_s in sorted_norms:
            pattern = r'\b(?:from\s+|supplier\s+|vendor\s+|documents\s+from\s+)?' + re.escape(norm_s) + r'\b'
            if re.search(pattern, norm_text):
                parsed.supplier = normalized_map[norm_s]
                words = norm_s.split()
                orig_pattern_str = r'\b(?:from\s+|supplier\s+|vendor\s+|documents\s+from\s+)?' + r'[\s\.,;:\-\'"]*'.join(re.escape(w) for w in words) + r'\b'
                orig_pattern = re.compile(orig_pattern_str, re.IGNORECASE)
                semantic_text = orig_pattern.sub(' ', semantic_text)
                break
    else:
        supplier_match = re.search(r'\b(?:from|supplier)\s+([A-Za-z0-9\s]+?)(?:\s+(?:below|under|over|above|for|amount)|$)', semantic_text, re.IGNORECASE)
        if supplier_match:
            parsed.supplier = supplier_match.group(1).strip()
            semantic_text = semantic_text[:supplier_match.start()] + semantic_text[supplier_match.end():]

    # Cleanup semantic text
    semantic_text = re.sub(r'\s+', ' ', semantic_text).strip()
    parsed.semantic_text = semantic_text

    return parsed
