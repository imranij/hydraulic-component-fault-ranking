# Deposit tooling

| file | what it is |
|---|---|
| `MANIFEST.csv` | path, size, SHA-256, role and redistribution status of every file in the release. The raw UCI files are listed with `deposited=no`: their digests are recorded, the files are not redistributed. |
| `build_manifest.py` | regenerates `MANIFEST.csv`; `--check` exits 1 if any deposited file changed, appeared or disappeared. |
| `check_deposit_ready.py` | exits 0 only when the manuscript carries the reserved DOI, the licence and version are set in `zenodo.json`, and the manifest matches the tree. |
| `zenodo.json` | Zenodo metadata for the record. |

Licence: code MIT; results, figures and text CC BY 4.0. Version 1.0.0 is the release that accompanies
the revised manuscript.
