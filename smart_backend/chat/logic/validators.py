import re

def validate_nid(nid):
    """Simple check for Ethiopian NID format (example only)"""
    return bool(re.fullmatch(r'\d{12}', nid))  # e.g., 12 digits

def validate_name(name):
    return bool(re.fullmatch(r'[A-Za-z\s]{2,50}', name.strip()))

def validate_email(email):
    return bool(re.fullmatch(r'[^@]+@[^@]+\.[^@]+', email))
