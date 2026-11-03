#!/usr/bin/env python3
r"""Refuse to call paper 09's deposit ready while any gate is open.

Exits 0 only when all four conditions hold:

  1. manuscript/main.tex defines \depositdoi as the reserved DOI (not the
     placeholder DEPOSITDOI);
  2. zenodo.json metadata.license is non-empty;
  3. zenodo.json metadata.version is non-empty;
  4. MANIFEST.csv matches the tree: every listed file still exists with the
     recorded SHA-256, and no deposited file is missing from the list.

Gate 4 is stricter than paper 10's, which only told you to run the manifest
check separately. It is stricter here for a specific reason: this project
shipped committed result CSVs that its own restored generator had not produced,
and a digest list that is never re-checked would not have caught it.

On success it also writes zenodo.upload.json: the same metadata with every
underscore-prefixed scaffolding key stripped, which is what Zenodo accepts.

usage:  python3 check_deposit_ready.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
MAIN = ROOT / "manuscript" / "main.tex"
ZEN = HERE / "zenodo.json"


def strip_under(o):
    if isinstance(o, dict):
        return {k: strip_under(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, list):
        return [strip_under(v) for v in o]
    return o


def main() -> int:
    fail = []

    t = MAIN.read_text(encoding="utf-8")
    m = re.search(r"\\newcommand\{\\depositdoi\}\{([^}]*)\}", t)
    if not m:
        fail.append(r"GATE 1  manuscript/main.tex defines no \depositdoi")
    elif m.group(1).strip() in ("", "DEPOSITDOI"):
        fail.append(r"GATE 1  \depositdoi is still the placeholder DEPOSITDOI")

    z = json.loads(ZEN.read_text(encoding="utf-8"))
    md = z["metadata"]
    if not str(md.get("license", "")).strip():
        fail.append("GATE 2  zenodo.json metadata.license is empty (no licence chosen; author decision)")
    if not str(md.get("version", "")).strip():
        fail.append("GATE 3  zenodo.json metadata.version is empty (fill with the release tag)")

    r = subprocess.run(
        [sys.executable, str(HERE / "build_manifest.py"), "--check"],
        capture_output=True, text=True,
    )
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        fail.append("GATE 4  MANIFEST.csv does not match the tree (see the lines above). "
                    "Re-run build_manifest.py only after confirming each change is intended.")

    if fail:
        print("\nDEPOSIT NOT READY:")
        for f in fail:
            print("  " + f)
        return 1

    up = HERE / "zenodo.upload.json"
    up.write_text(json.dumps(strip_under(z), indent=2, ensure_ascii=False) + "\n",
                  encoding="utf-8", newline="")
    print(f"\nALL GATES CLEAR -- wrote {up.name}")
    return 0


raise SystemExit(main())
