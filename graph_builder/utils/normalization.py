import re
from datetime import datetime
from graph_builder.schema.json_models import CandidateTriple

def normalize_date(date_str: str) -> str:
    """
    Normalizes various date formats to ISO 8601 (YYYY-MM-DD).
    """
    if not date_str:
        return ""
    try:
        # Handles YYYY, YYYY-MM, YYYY-MM-DD
        if re.match(r'^\d{4}$', date_str):
            return f"{date_str}-01-01" # Assume start of year
        if re.match(r'^\d{4}-\d{2}$', date_str):
            return f"{date_str}-01" # Assume start of month
        dt = datetime.fromisoformat(date_str)
        return dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return date_str # Return original if parsing fails

def normalize_triple(tri: CandidateTriple) -> CandidateTriple:
    """
    Applies various normalization routines to a CandidateTriple.
    """
    tri.head = tri.head.strip().title()
    tri.tail = tri.tail.strip().title()
    
    if tri.as_of_date:
        tri.as_of_date = normalize_date(tri.as_of_date)
    if tri.start_time:
        tri.start_time = normalize_date(tri.start_time)
    if tri.end_time:
        tri.end_time = normalize_date(tri.end_time)
        
    # Add other normalizations here:
    # - URL canonicalization
    # - Geographic name standardization (e.g., "NYC" -> "New York City")
    # - Identifier validation (e.g., DOI, ISIN)
    
    return tri
from datetime import datetime
from graph_builder.schema.json_models import CandidateTriple

def normalize_date(date_str: str) -> str:
    """
    Normalizes various date formats to ISO 8601 (YYYY-MM-DD).
    """
    if not date_str:
        return ""
    try:
        # Handles YYYY, YYYY-MM, YYYY-MM-DD
        if re.match(r'^\d{4}$', date_str):
            return f"{date_str}-01-01" # Assume start of year
        if re.match(r'^\d{4}-\d{2}$', date_str):
            return f"{date_str}-01" # Assume start of month
        dt = datetime.fromisoformat(date_str)
        return dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return date_str # Return original if parsing fails

def normalize_triple(tri: CandidateTriple) -> CandidateTriple:
    """
    Applies various normalization routines to a CandidateTriple.
    """
    tri.head = tri.head.strip().title()
    tri.tail = tri.tail.strip().title()
    
    if tri.as_of_date:
        tri.as_of_date = normalize_date(tri.as_of_date)
    if tri.start_time:
        tri.start_time = normalize_date(tri.start_time)
    if tri.end_time:
        tri.end_time = normalize_date(tri.end_time)
        
    # Add other normalizations here:
    # - URL canonicalization
    # - Geographic name standardization (e.g., "NYC" -> "New York City")
    # - Identifier validation (e.g., DOI, ISIN)
    
    return tri
from datetime import datetime
from graph_builder.schema.json_models import CandidateTriple

def normalize_date(date_str: str) -> str:
    """
    Normalizes various date formats to ISO 8601 (YYYY-MM-DD).
    """
    if not date_str:
        return ""
    try:
        # Handles YYYY, YYYY-MM, YYYY-MM-DD
        if re.match(r'^\d{4}$', date_str):
            return f"{date_str}-01-01" # Assume start of year
        if re.match(r'^\d{4}-\d{2}$', date_str):
            return f"{date_str}-01" # Assume start of month
        dt = datetime.fromisoformat(date_str)
        return dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return date_str # Return original if parsing fails

def normalize_triple(tri: CandidateTriple) -> CandidateTriple:
    """
    Applies various normalization routines to a CandidateTriple.
    """
    tri.head = tri.head.strip().title()
    tri.tail = tri.tail.strip().title()
    
    if tri.as_of_date:
        tri.as_of_date = normalize_date(tri.as_of_date)
    if tri.start_time:
        tri.start_time = normalize_date(tri.start_time)
    if tri.end_time:
        tri.end_time = normalize_date(tri.end_time)
        
    # Add other normalizations here:
    # - URL canonicalization
    # - Geographic name standardization (e.g., "NYC" -> "New York City")
    # - Identifier validation (e.g., DOI, ISIN)
    
    return tri
