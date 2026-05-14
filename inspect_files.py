"""
inspect_files.py
================
Auto-finds MIMIC-IV and eICU files (.csv or .csv.gz) and prints
headers + sample rows. Works on compressed files without extracting them.

Usage:
    python inspect_files.py
"""

import os
import csv
import gzip

def find_file(filename, search_root='.'):
    """
    Search recursively for filename or filename.gz.
    Returns (path, is_gzipped).
    """
    for root, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs
                   if not d.startswith('.') and d != '__pycache__']
        if filename in files:
            return os.path.join(root, filename), False
        if filename + '.gz' in files:
            return os.path.join(root, filename + '.gz'), True
    return None, False


def inspect_file(filename, search_root='.', n_rows=3):
    """Read header + n_rows from a CSV or CSV.GZ file."""
    path, is_gz = find_file(filename, search_root)

    if path is None:
        print(f"  NOT FOUND: {filename} or {filename}.gz")
        print()
        return None, None

    size_mb  = os.path.getsize(path) / (1024**2)
    size_str = (f"{size_mb:.0f} MB" if size_mb < 1000
                else f"{size_mb/1024:.1f} GB")
    fmt = "gzip CSV" if is_gz else "CSV"
    print(f"  Found: {path}  ({size_str}, {fmt})")

    rows = []
    try:
        opener = gzip.open(path, 'rt', encoding='utf-8',
                           errors='replace') if is_gz else \
                 open(path, encoding='utf-8', errors='replace')
        with opener as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= n_rows:
                    break
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        print()
        return None, None

    if not rows:
        print("  EMPTY FILE")
        print()
        return None, None

    headers = rows[0]
    data    = rows[1:]
    print(f"  Columns ({len(headers)}): {headers}")
    if data:
        print(f"  Row 1: {data[0]}")
    print()
    return headers, data


# ── Main ──────────────────────────────────────────────────────

print("=" * 65)
print("FILE INSPECTOR — FuzzyBoost Clinical Data")
print("=" * 65)
print(f"Searching from: {os.path.abspath('.')}")
print()

# --- MIMIC-IV ---
print("--- MIMIC-IV ---")
print()

mimic_files = [
    'admissions.csv',
    'patients.csv',
    'labevents.csv',
    'icustays.csv',
    'chartevents.csv',
]

mimic_headers = {}
for fname in mimic_files:
    h, _ = inspect_file(fname)
    mimic_headers[fname] = h

# --- eICU ---
print("--- eICU ---")
print()

eicu_files = [
    'patient.csv',
    'lab.csv',
    'vitalPeriodic.csv',
]

eicu_headers = {}
for fname in eicu_files:
    h, _ = inspect_file(fname)
    eicu_headers[fname] = h

# --- Summary ---
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print()
print("MIMIC-IV:")
for fname, h in mimic_headers.items():
    status = f"OK  ({len(h)} cols)" if h else "MISSING"
    print(f"  {fname:<30} {status}")

print()
print("eICU:")
for fname, h in eicu_headers.items():
    status = f"OK  ({len(h)} cols)" if h else "MISSING"
    print(f"  {fname:<30} {status}")

print()
print("Send the complete output of this script to Avoy.")
