#!/usr/bin/env python3
"""Regenerate MANIFEST.csv for paper 09's Zenodo deposit.

Every row is measured from a file that exists in this project right now:
path relative to the project root, byte size, SHA-256, role, and whether the
file is redistributed in the deposit.  Nothing here is estimated.

Raw third-party inputs are listed with deposited=no.  Their digests are
recorded so a third party can check that the copy they obtain from the
original source is the copy this study used; redistributing them is a
licence question and therefore an author decision, not a build step.

usage:  python3 build_manifest.py          # writes MANIFEST.csv
        python3 build_manifest.py --check  # verifies MANIFEST.csv, exit 1 on drift
"""
import csv, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "MANIFEST.csv"

# (glob, role, deposited, note)
RULES = [
    ("code/*.py",                        "pipeline, revision and verifier code", "yes", ""),
    ("code/requirements.txt",            "environment pin",             "yes", ""),
    ("code/EMITTER_ORDER.md",            "documentation",               "yes", ""),
    ("code/PROVENANCE_REPORT.txt",       "verifier output",             "yes", "submission-version manuscript"),
    ("results/*.csv",                    "result record",               "yes", ""),
    ("results/*.md",                     "provenance note",             "yes", "retirement notes for superseded results; retained, not deleted"),
    ("results/_preexisting_20260917/*",  "superseded result record",    "yes", "kept for audit"),
    ("results/revision/*.csv",           "result record (revision)",    "yes", ""),
    ("results/revision/*.json",          "result record (revision)",    "yes", ""),
    ("results/revision/cache/*.npz",     "cached out-of-fold predictions (revision)", "yes", ""),
    ("data/processed/*.csv",             "derived feature table",       "yes", "derived from the UCI hydraulic raw streams"),
    ("data/processed/*.json",            "derived feature table",       "yes", ""),
    ("figures/*.pdf",                    "figure",                      "yes", ""),
    ("manuscript/main.tex",              "manuscript source",           "yes", "revised manuscript"),
    ("manuscript/main.bbl",              "manuscript source",           "yes", ""),
    ("manuscript/references.bib",        "manuscript source",           "yes", ""),
    ("manuscript/generated_*.tex",       "generated table",             "yes", ""),
    ("README.md",                        "documentation",               "yes", ""),
    ("HOW_TO_INSTALL_AND_REPRODUCE.md",  "documentation",               "yes", ""),
    ("LICENSE",                          "licence",                     "yes", ""),
    ("CITATION.cff",                     "citation metadata",           "yes", ""),
    ("deposit/*.py",                     "deposit tooling",             "yes", ""),
    ("deposit/zenodo.json",              "deposit metadata",            "yes", ""),
    ("deposit/README.md",                "documentation",               "yes", ""),
    ("data/uci_hydraulic_raw/*.txt",     "raw third-party input",       "no",  "UCI Condition Monitoring of Hydraulic Systems (Helwig, Pignanelli & Schutze, I2MTC 2015, doi:10.1109/I2MTC.2015.7151267), via https://archive.ics.uci.edu/ml"),
]

EXCLUDE_SUBSTR = ("__pycache__", ".bak", ".RETIRED.")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    rows, seen = [], set()
    for pat, role, dep, note in RULES:
        hits = sorted(ROOT.glob(pat))
        if not hits:
            print(f"WARNING  no file matches {pat}", file=sys.stderr)
        for p in hits:
            rel = p.relative_to(ROOT).as_posix()
            if not p.is_file() or rel in seen:
                continue
            if any(s in rel for s in EXCLUDE_SUBSTR):
                continue
            seen.add(rel)
            rows.append({"path": rel, "bytes": p.stat().st_size,
                         "sha256": sha256(p), "role": role,
                         "deposited": dep, "note": note})
    rows.sort(key=lambda r: r["path"])
    return rows


def main():
    rows = collect()
    check = "--check" in sys.argv
    if check:
        if not OUT.exists():
            print("MANIFEST.csv absent"); return 1
        old = list(csv.DictReader(OUT.open(encoding="utf-8")))
        oldmap = {r["path"]: r for r in old}
        newmap = {r["path"]: r for r in rows}
        bad = 0
        for path in sorted(set(oldmap) | set(newmap)):
            if path not in oldmap:
                print(f"NEW      {path}"); bad += 1
            elif path not in newmap:
                if oldmap[path]["deposited"] == "no":
                    continue  # third-party raw input: digest recorded, file not redistributed
                print(f"MISSING  {path}"); bad += 1
            elif oldmap[path]["sha256"] != newmap[path]["sha256"]:
                print(f"CHANGED  {path}"); bad += 1
        print(f"{len(rows)} files, {bad} discrepancies")
        return 1 if bad else 0
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "role", "deposited", "note"])
        w.writeheader()
        w.writerows(rows)
    dep = sum(1 for r in rows if r["deposited"] == "yes")
    print(f"wrote MANIFEST.csv  {len(rows)} files  ({dep} deposited, {len(rows)-dep} withheld)")
    return 0


raise SystemExit(main())
