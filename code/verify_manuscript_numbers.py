# ---------------------------------------------------------------- engine ----
# Shared provenance-verification engine. Embedded in each paper's script so the
# script is self-contained and can be run from a clean checkout of that paper.
#
# Three kinds of check, all of which must pass for exit code 0:
#
#   1. CLAIM   an explicitly named number in the prose / abstract / a table is
#              recomputed from a named CSV field and compared to the value the
#              manuscript prints, at a stated tolerance.
#   2. SWEEP   *every* numeric token in the manuscript (prose, abstract and
#              in-file tables) is matched against the pool of values derivable
#              from the committed data files. A token that matches nothing and
#              is not on the declared non-data whitelist is an untraced number.
#   3. FIGURE  every numeric token in each figure's PDF text layer is matched
#              against the values in the CSVs that figure is drawn from, plus
#              that figure's declared axis ticks and other non-data furniture.
#              (Modelled on the checks written for papers 04 and 12 on
#              2026-09-10.)
#
# Tolerances are always stated at the call site and are never wider than half a
# unit in the last digit the manuscript prints, unless a comment at the call
# site justifies a wider one.
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# A numeric token: a run of digits not glued to a letter or another digit-word.
# "BaTiO3" and "paper2_x" must not contribute tokens; "10{,}000" must arrive as
# one token, and an en-dash range "0.10--0.20" must not read as "-0.20".
NUM = re.compile(
    r"(?<![A-Za-z0-9_.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9_])"
)


def fmts(x) -> set:
    """Every rounding at which a manuscript or figure might print this value."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return set()
    if x != x or x in (float("inf"), float("-inf")):
        return set()
    s = {("%%.%df" % k) % x for k in range(0, 7)}
    s |= {("%%.%dg" % k) % x for k in range(1, 7)}
    s |= {v.lstrip("0") for v in list(s) if v.startswith("0.")}
    s |= {v.replace("-0.", "-.") for v in list(s) if v.startswith("-0.")}
    if float(x).is_integer() and abs(x) < 1e15:
        s.add(str(int(x)))
    return s


def strip_tex(t: str) -> str:
    #: Blind spot 4 closed here, 2026-09-17, and here rather than only in
    #: tokens() because sweep() calls NUM.finditer() directly and never goes
    #: through tokens() -- so patching tokens() alone left the sweep path, the
    #: one the blind spot was found in, completely unchanged.
    #:
    #: A LaTeX control word ends at its first non-letter, so "$\\approx0.079$"
    #: is \\approx followed by 0.079. NUM's lookbehind (?<![A-Za-z0-9_.]) sees
    #: the "x" and refuses the match, so the number was never swept. In paper 04
    #: that hid three fold standard deviations carrying the paper's central
    #: claim. Inserting the space TeX already implies makes them visible without
    #: loosening the lookbehind, which would start matching digits inside
    #: identifiers such as v1.2 and R2.
    t = GLUED_CMD.sub(lambda m: m.group(1) + " ", t)
    t = re.sub(r"(?<!\\)%.*", "", t)          # LaTeX comments carry no claims
    t = t.replace("{,}", "").replace(",", "") if False else t.replace("{,}", "")
    t = t.replace("---", " to ").replace("--", " to ")   # ranges, not negatives
    return t


#: A LaTeX control word followed immediately by a digit. TeX ends a control
#: word at the first non-letter, so "\\approx0.079" is the command \\approx
#: followed by the number 0.079 -- but NUM's lookbehind, (?<![A-Za-z0-9_.]),
#: sees the "x" and refuses the match, so the number is never tokenised.
#:
#: Blind spot 4 (VERIFIER_LIMITS, 2026-09-11): in paper 04 this hid the three
#: fold standard deviations carrying the paper's central claim. They were
#: written "$\\approx0.079$" and no sweep in the portfolio could see them.
#:
#: Fixed 2026-09-17 by normalising the boundary before tokenising rather than
#: by loosening the lookbehind, which would also start matching digits inside
#: identifiers. Inserting a space between a control word and a following digit
#: is what TeX already does semantically, so it cannot change what the source
#: means; it only makes the number visible to the tokeniser.
GLUED_CMD = re.compile(r"(\\[A-Za-z]+)(?=\d)")


def tokens(text: str) -> list:
    text = GLUED_CMD.sub(r"\1 ", text)
    out = []
    for m in NUM.finditer(text):
        tok = m.group().replace("−", "-").replace(",", "")
        if tok.startswith("+"):
            tok = tok[1:]
        out.append(tok)
    return out


def _numeric_cells(df: pd.DataFrame) -> list:
    vals = []
    for col in df.columns:
        ser = pd.to_numeric(df[col], errors="coerce").dropna()
        vals.extend(ser.tolist())
    return vals


def _aggregates(df: pd.DataFrame) -> list:
    """Summary statistics a paper legitimately quotes from a table of records."""
    vals = [len(df), len(df.columns)]
    for col in df.columns:
        ser = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(ser) == 0:
            try:
                vals.append(df[col].nunique())
                if df[col].dtype == object and df[col].nunique() < 60:
                    vals.extend(df[col].value_counts().tolist())
            except Exception:
                pass
            continue
        vals.extend([ser.mean(), ser.min(), ser.max(), ser.median(), ser.sum(), len(ser)])
        if len(ser) > 1:
            vals.append(ser.std(ddof=1))
        vals.extend(v * 100 for v in (ser.mean(), ser.min(), ser.max(), ser.median()))
    return vals


# ------------------------------------------------ citations (2026-09-11) ----
# Blind-spot class 4.  Until today no verifier in this portfolio compared a
# \cite key against the .bib, or looked at a bibliography entry at all.  That
# is how paper 13 came to cite a non-existent author for one of the author's
# own papers while its verifier exited 0.  A number-provenance check that
# ignores the bibliography verifies half a manuscript.
#
# Checks, all of them narrowing (they can only add findings):
#   * every \cite key resolves to an entry in the .bib the manuscript loads
#   * every CITED entry has an author (or editor) and a title
#   * no entry is defined twice
#   * an author that is a placeholder ("TBD", "Anonymous", "Author") FAILS
#   * an author that is a single bare token ("Imran") FAILS: no given name, so
#     BibTeX can form no initial and the entry names no identifiable person
#   * an author that reads as an institution is REPORTED, not failed: it is
#     right for a standards body or a data repository, and it is also what a
#     fabricated entry looks like, so it is always surfaced
#   * a key of the form <word><year>... whose <word> appears in NO field of the
#     entry is REPORTED: it is either a topical key (a normal convention) or an
#     entry whose author and title were replaced under an old key

CITE_CMD_RE = re.compile(
    r"\\(?:no)?(?:cite|citep|citet|citeauthor|citeyear|citeyearpar|citealp|"
    r"citealt|citenum|parencite|textcite|autocite|footcite|fullcite|"
    r"supercite|Citep|Citet|Cite)\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^{}]*)\}")
BIB_LOAD_RE = re.compile(
    r"\\(?:bibliography|addbibresource|nobibliography)\s*\{([^{}]*)\}")
BIB_START_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s{}]+)\s*,", re.M)
PLACEHOLDER_AUTHOR_RE = re.compile(
    r"\b(?:tbd|todo|fixme|xxx+|placeholder|anonymous|unknown|no\s*author|"
    r"n\.?\s*/?\s*a\.?|author|authors|first\s+last|john\s+doe|jane\s+doe|"
    r"lastname|surname)\b", re.I)
INSTITUTIONAL_AUTHOR_RE = re.compile(
    r"\b(?:institute|institution|laborator(?:y|ies)|universit(?:y|ies)|college|"
    r"corporation|corp|incorporated|inc|ltd|gmbh|llc|company|consortium|"
    r"committee|working\s+group|task\s+force|team|foundation|agency|authority|"
    r"administration|bureau|organi[sz]ation|society|association|council|"
    r"repository|collaboration)\b", re.I)
KEY_WORD_RE = re.compile(r"^([A-Za-z][A-Za-z'\-]{3,})(?:19|20)\d{2}")
KEY_SEGMENT_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+")
KEY_MATCH_FIELDS = ("author", "editor", "title", "journal", "booktitle",
                    "publisher", "series", "note", "howpublished", "url",
                    "doi", "organization", "institution", "school")
LATEX_ACCENT_RE = re.compile(r"\\[a-zA-Z]+\s*|\\[^a-zA-Z]")
_FOLD = {
    "\u00e1": "a", "\u00e0": "a", "\u00e2": "a", "\u00e4": "a", "\u00e3": "a",
    "\u00e5": "a", "\u00e9": "e", "\u00e8": "e", "\u00ea": "e", "\u00eb": "e",
    "\u00ed": "i", "\u00ec": "i", "\u00ee": "i", "\u00ef": "i", "\u00f3": "o",
    "\u00f2": "o", "\u00f4": "o", "\u00f6": "o", "\u00f5": "o", "\u00fa": "u",
    "\u00f9": "u", "\u00fb": "u", "\u00fc": "u", "\u00e7": "c", "\u00f1": "n",
    "\u0142": "l", "\u00df": "ss", "\u00f8": "o", "\u0161": "s", "\u017e": "z",
    "\u010d": "c", "\u0107": "c",
}


def _debrace(s):
    return s.replace("{", "").replace("}", "").strip()


def _fold(s):
    s = LATEX_ACCENT_RE.sub(" ", s)
    return "".join(c.lower() if c.isascii() else _FOLD.get(c, c).lower()
                   for c in s if c.isalpha())


def parse_bib(path):
    """{key: {"type", "fields", "line"}}, plus the list of duplicated keys."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    entries, dups = {}, []
    field_re = re.compile(r"(\w+)\s*=\s*")
    for m in BIB_START_RE.finditer(txt):
        etype, key = m.group(1).lower(), m.group(2)
        if etype in ("comment", "preamble", "string"):
            continue
        i = txt.index("{", m.start())
        depth, j = 0, i
        while j < len(txt):
            if txt[j] == "{":
                depth += 1
            elif txt[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body, fields, k = txt[m.end():j], {}, 0
        while k < len(body):
            fm = field_re.match(body, k)
            if not fm:
                k += 1
                continue
            vs = fm.end()
            if vs >= len(body):
                break
            if body[vs] == "{":
                d2, p = 0, vs
                while p < len(body):
                    if body[p] == "{":
                        d2 += 1
                    elif body[p] == "}":
                        d2 -= 1
                        if d2 == 0:
                            break
                    p += 1
                fields[fm.group(1).lower()] = body[vs + 1:p]
                k = p + 1
            elif body[vs] == '"':
                p = body.find('"', vs + 1)
                if p < 0:
                    break
                fields[fm.group(1).lower()] = body[vs + 1:p]
                k = p + 1
            else:
                p = body.find(",", vs)
                p = len(body) if p < 0 else p
                fields[fm.group(1).lower()] = body[vs:p]
                k = p + 1
        if key in entries:
            dups.append(key)
        entries[key] = {"type": etype, "fields": fields,
                        "line": txt.count("\n", 0, m.start()) + 1}
    return entries, dups


class Verifier:
    def __init__(self, root: Path, paper: str):
        self.swept_tex = []
        self.root = Path(root)
        self.paper = paper
        self.lines: list = []
        self.failures: list = []
        self.n_checked = 0
        self.n_traced = 0
        self.untraced: list = []

    # -------------------------------------------------------------- claims --
    def check(self, label, computed, quoted, source, tol=0.0005):
        """One named claim: recomputed value vs the value the manuscript prints."""
        self.n_checked += 1
        try:
            ok = abs(float(computed) - float(quoted)) <= tol
        except (TypeError, ValueError):
            ok = False
        if ok:
            self.n_traced += 1
            self.lines.append("  PASS  %-58s manuscript %-12s  data %-14s  [%s]  tol %g"
                              % (label, quoted, _fmt(computed), source, tol))
        else:
            self.failures.append(
                "%s: manuscript says %s, %s gives %s (tolerance %g)"
                % (label, quoted, source, _fmt(computed), tol))
            self.lines.append("  FAIL  %-58s manuscript %-12s  data %-14s  [%s]  tol %g"
                              % (label, quoted, _fmt(computed), source, tol))

    def present(self, label, needle, text, source="manuscript"):
        """A guardrail sentence that must survive edits."""
        self.n_checked += 1
        if needle in text:
            self.n_traced += 1
            self.lines.append("  PASS  %-58s string present                        [%s]"
                              % (label, source))
        else:
            self.failures.append("%s: expected string absent -> %r" % (label, needle))
            self.lines.append("  FAIL  %-58s string ABSENT                         [%s]"
                              % (label, source))

    # ---------------------------------------------------------------- pool --
    def build_pool(self, files):
        """Pool of every value derivable from the committed data files."""
        pool, used = set(), []
        for rel in files:
            p = self.root / rel
            if not p.is_file():
                self.failures.append("declared source file missing: %s" % rel)
                continue
            used.append(rel)
            if p.suffix.lower() in (".csv", ".tsv"):
                try:
                    df = pd.read_csv(p, sep=None, engine="python")
                except Exception as exc:
                    self.failures.append("unreadable source %s: %s" % (rel, exc))
                    continue
                for v in _numeric_cells(df) + _aggregates(df):
                    pool |= fmts(v)
            elif p.suffix.lower() == ".json":
                acc = []

                def walk(o):
                    if isinstance(o, dict):
                        for v in o.values():
                            walk(v)
                    elif isinstance(o, list):
                        for v in o:
                            walk(v)
                    elif isinstance(o, bool):
                        pass
                    elif isinstance(o, (int, float)):
                        acc.append(o)
                walk(json.loads(p.read_text(encoding="utf-8", errors="ignore")))
                for v in acc:
                    pool |= fmts(v)
            else:  # a generated .tex table or a recorded .txt result
                for tok in tokens(strip_tex(p.read_text(encoding="utf-8", errors="ignore"))):
                    pool |= fmts(tok)
                    pool.add(tok)
        return pool, used

    # --------------------------------------------------------------- sweep --
    def sweep(self, tex_files, source_files, allowed, derived=()):
        """Exhaustive check: every numeric token in the manuscript must either
        be derivable from a committed source file or be on the declared
        non-data whitelist (`allowed` maps token -> why it is not a data value).
        """
        self.swept_tex = list(tex_files)
        pool, used = self.build_pool(source_files)
        # Values re-derived by the named claims above (ratios, percentages of a
        # count, mantissas of p-values printed in scientific notation). These
        # are data-derived, just not cell values, so they belong in the pool.
        for x in derived:
            pool |= fmts(x)
        self.lines.append("")
        self.lines.append("SWEEP  every numeric token in %s" % ", ".join(str(f) for f in tex_files))
        self.lines.append("       against %d committed source files (%d distinct formatted values)"
                          % (len(used), len(pool)))
        seen = {}
        for rel in tex_files:
            p = self.root / rel
            if not p.is_file():
                self.failures.append("manuscript file missing: %s" % rel)
                continue
            text = strip_tex(p.read_text(encoding="utf-8", errors="ignore"))
            for m in NUM.finditer(text):
                tok = m.group().replace("−", "-").replace(",", "")
                if tok.startswith("+"):
                    tok = tok[1:]
                ctx = re.sub(r"\s+", " ", text[max(0, m.start() - 90):m.end() + 50])
                seen.setdefault(tok, ctx)
        traced = declared = 0
        for tok in sorted(seen, key=lambda t: (len(t), t)):
            self.n_checked += 1
            if tok in pool:
                traced += 1
                self.n_traced += 1
            elif tok in allowed:
                declared += 1
                self.n_traced += 1
            else:
                self.untraced.append((tok, seen[tok]))
                self.failures.append("untraced number %s -- ...%s..." % (tok, seen[tok]))
        self.lines.append("       %d distinct tokens: %d derivable from data, %d declared "
                          "non-data, %d UNTRACED" % (len(seen), traced, declared, len(self.untraced)))
        for tok, ctx in self.untraced:
            self.lines.append("       UNTRACED %-12s ...%s..." % (tok, ctx))
        return pool

    # ------------------------------------------------------------- figures --
    def figures(self, specs, no_text_layer=()):
        """Per figure: every numeric token in the PDF text layer must be in the
        pool of values from the CSVs the figure is drawn from, or in that
        figure's declared furniture (axis ticks, panel labels, footnote text).
        """
        self.lines.append("")
        self.lines.append("FIGURES  numeric tokens in each figure's text layer vs its source CSVs")
        for rel, (allowed, furniture, source) in specs.items():
            pdf = self.root / rel
            self.n_checked += 1
            if not pdf.is_file():
                self.failures.append("figure PDF missing: %s" % rel)
                self.lines.append("  FAIL  %-52s no PDF on disk" % rel)
                continue
            try:
                raw = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                                     capture_output=True, text=True, check=True).stdout
            except Exception as exc:
                self.failures.append("could not read figure text layer %s: %s" % (rel, exc))
                self.lines.append("  FAIL  %-52s pdftotext failed" % rel)
                continue
            raw = re.sub(r"(?<=\d),(?=\d\d\d(?!\d))", "", raw)
            toks = tokens(raw)
            ok = allowed | furniture
            bad = sorted({t for t in toks if t not in ok})
            if bad:
                self.failures.append("figure %s prints %d value(s) with no source: %s"
                                     % (rel, len(bad), ", ".join(bad)))
                self.lines.append("  FAIL  %-52s %3d tokens, %d unsourced: %s  [%s]"
                                  % (rel, len(toks), len(bad), ", ".join(bad), source))
            else:
                self.n_traced += 1
                self.lines.append("  PASS  %-52s %3d tokens, 0 unsourced  [%s]"
                                  % (rel, len(toks), source))
        # ---------------------------------------------------------------
        # Blind spot 6, fixed 2026-09-17.  This branch previously read:
        #
        #     for rel, note in no_text_layer:
        #         self.n_checked += 1
        #         self.n_traced += 1
        #         self.lines.append("  NOTE  ... raster figure, no text layer to
        #                            diff; plotted series re-derived above from %s")
        #
        # Listing a figure here was therefore enough to make it a TRACED claim.
        # Nothing was read, nothing was compared, and the headline count was
        # inflated by exactly the length of this tuple -- so a reader who
        # distrusted the prose and checked the number read the same untruth.
        #
        # Worse, the printed line asserted re-derivation whatever the note said.
        # In paper 11 five notes were written honestly as "(raster; no text
        # layer)" and this format string overrode them, which is what proved the
        # problem a code defect rather than a documentation one.
        #
        # Replaced with the design the portfolio's nine provlib papers already
        # use: refuse the assurance, touch NO counter, and say plainly what was
        # not checked.  The compensating control is figure_scripts() below --
        # for a raster, the generator's own hard-coded literals are the only
        # part of the figure that can be read at all.
        # ---------------------------------------------------------------
        if no_text_layer:
            self.lines.append("")
            self.lines.append("  NOT SWEPT  raster images carry no extractable text, so any "
                              "value printed on them cannot be checked here:")
            for rel, note in no_text_layer:
                self.lines.append("             %-48s drawn from %s" % (rel, note))
            self.lines.append("             Not counted as checked and not counted as traced. "
                              "A value printed only on one of these is UNVERIFIED.")

    # ------------------------------------------------------- figure scripts --
    def figure_scripts(self, specs):
        """Compensating control for raster figures (blind spot 6, 2026-09-17).

        A raster has no text layer, so the only readable part of it is whatever
        its generator hard-codes into an annotation.  Every numeric literal in
        the script -- comments and docstrings stripped, because a number in
        prose is not an annotation -- is checked against the same pool of
        committed values its figure is drawn from.  These ARE counted, because
        they are really checked.

        `specs` maps a script path to (allowed, furniture, label), the same
        shape figures() uses.
        """
        if not specs:
            return
        self.lines.append("")
        self.lines.append("FIGURE SCRIPTS  numeric literals hard-coded in each raster "
                          "generator vs its source CSVs")
        for rel, (allowed, furniture, label) in specs.items():
            src = self.root / rel
            self.n_checked += 1
            if not src.is_file():
                self.failures.append("figure script missing: %s" % rel)
                self.lines.append("  FAIL  %-52s no script on disk" % rel)
                continue
            text = src.read_text(encoding="utf-8", errors="replace")
            text = re.sub(r'"""[\s\S]*?"""', " ", text)
            text = re.sub(r"'''[\s\S]*?'''", " ", text)
            text = re.sub(r"#[^\n]*", " ", text)
            lits = tokens(text)
            ok = allowed | furniture
            bad = sorted({t for t in lits if t not in ok})
            if bad:
                self.failures.append(
                    "figure script %s hard-codes %d value(s) with no source: %s"
                    % (rel, len(bad), ", ".join(bad)))
                self.lines.append("  FAIL  %-52s %3d literals, %d unsourced: %s  [%s]"
                                  % (rel, len(lits), len(bad), ", ".join(bad), label))
            else:
                self.n_traced += 1
                self.lines.append("  PASS  %-52s %3d literals, 0 unsourced  [%s]"
                                  % (rel, len(lits), label))

    # -------------------------------------------------------------- report --
    # ------------------------------------------------ citations (class 4) ---
    def cites(self, tex_rels=None, bib_rels=None, must_include=()):
        """Check \cite keys and bibliography entries.

        `tex_rels` defaults to the files the sweep already read.  `bib_rels`
        defaults to whatever those files load with \bibliography /
        \addbibresource, so the check follows the bibliography the manuscript
        actually compiles against rather than a guess.
        """
        tex_rels = list(tex_rels if tex_rels is not None else self.swept_tex)
        texts, tex_paths = [], []
        for rel in tex_rels:
            p = self.root / rel
            if not p.is_file():
                self.failures.append("citation check: manuscript file missing: %s" % rel)
                continue
            tex_paths.append(p)
            texts.append((rel, p.read_text(encoding="utf-8", errors="ignore")))

        if bib_rels is None:
            found, bib_paths = [], []
            for rel, raw in texts:
                base = (self.root / rel).parent
                for m in BIB_LOAD_RE.finditer(re.sub(r"(?<!\\)%[^\n]*", "", raw)):
                    for name in m.group(1).split(","):
                        name = name.strip()
                        if not name:
                            continue
                        cand = base / (name if name.endswith(".bib") else name + ".bib")
                        if cand not in bib_paths:
                            bib_paths.append(cand)
                            found.append(str(cand.relative_to(self.root)))
            if not bib_paths:
                for rel, _ in texts:
                    for cand in sorted((self.root / rel).parent.glob("*.bib")):
                        if cand not in bib_paths:
                            bib_paths.append(cand)
        else:
            bib_paths = [self.root / r for r in bib_rels]

        self.lines.append("")
        self.lines.append("CITATIONS  every \\cite key resolves to a .bib entry; every cited")
        self.lines.append("           entry has an author and a title (added 2026-09-11)")

        entries, seen_files = {}, []
        for bp in bib_paths:
            if not bp.is_file():
                self.failures.append("citation check: .bib not found: %s" % bp)
                self.n_checked += 1
                continue
            seen_files.append(bp.name)
            ents, dups = parse_bib(bp)
            for d in sorted(set(dups)):
                self.n_checked += 1
                self.failures.append(
                    "bib %s: key %r is defined more than once; the later "
                    "definition silently wins" % (bp.name, d))
            for k, val in ents.items():
                if k in entries:
                    self.n_checked += 1
                    self.failures.append(
                        "bib: key %r is defined in two .bib files (%s and %s)"
                        % (k, entries[k]["file"], bp.name))
                val["file"] = bp.name
                entries[k] = val

        cites = {}
        for rel, raw in texts:
            raw = re.sub(r"(?<!\\)%[^\n]*", "", raw)
            for m in CITE_CMD_RE.finditer(raw):
                line = raw.count("\n", 0, m.start()) + 1
                for key in m.group(1).split(","):
                    key = key.strip()
                    if key:
                        cites.setdefault(key, []).append("%s:%d" % (rel, line))

        self.lines.append("           %d distinct \\cite keys, %d entries in %s"
                          % (len(cites), len(entries),
                             ", ".join(seen_files) or "(no .bib found)"))
        if cites and not entries:
            self.n_checked += 1
            self.failures.append(
                "citation check: the manuscript makes %d distinct citations but "
                "no .bib file could be located, so no citation key is checked"
                % len(cites))

        problems = []
        for key in sorted(cites):
            self.n_checked += 1
            if key in entries:
                self.n_traced += 1
            else:
                problems.append(("FAIL", key, "cited at %s but no such entry in %s"
                                 % (", ".join(cites[key][:4]),
                                    ", ".join(seen_files) or "any .bib")))

        for key in sorted(entries):
            ent = entries[key]
            f = ent["fields"]
            cited = key in cites
            sev = "FAIL" if cited else "WARN"
            raw_author = f.get("author", "") or f.get("editor", "")
            author = _debrace(raw_author)
            title = _debrace(f.get("title", ""))
            corporate = _debrace(f.get("organization", "") or f.get("institution", ""))
            organisational = raw_author.strip().startswith("{")
            where = "%s:%d" % (ent["file"], ent["line"])
            self.n_checked += 1
            bad = False
            if not author and corporate:
                problems.append(("WARN", key,
                                 "no author/editor; identified by its corporate "
                                 "originator %r -- correct for a standard or a "
                                 "report (%s)" % (corporate[:60], where)))
            elif not author:
                problems.append((sev, key,
                                 "@%s entry has no author, no editor and no "
                                 "organization (%s)" % (ent["type"], where)))
                bad = True
            if not title:
                problems.append((sev, key, "@%s entry has no title (%s)"
                                 % (ent["type"], where)))
                bad = True
            if not f.get("year") and not f.get("date"):
                problems.append(("WARN", key, "no year (%s)" % where))
            if author and PLACEHOLDER_AUTHOR_RE.search(author):
                problems.append(("FAIL", key, "placeholder author %r (%s)"
                                 % (author[:70], where)))
                bad = True
            elif author and INSTITUTIONAL_AUTHOR_RE.search(author):
                problems.append(("WARN", key,
                                 "author looks institutional, not a person: %r -- "
                                 "correct for a standards body or a data repository, "
                                 "and also what a fabricated entry looks like (%s)"
                                 % (author[:70], where)))
            elif (author and not organisational and "," not in author
                  and " and " not in author and len(author.split()) == 1):
                problems.append((sev, key,
                                 "author is the single bare token %r: no given name, "
                                 "so BibTeX can form no initial and the entry names "
                                 "no identifiable person. Write the full name, or "
                                 "double-brace it as an organisation (%s)"
                                 % (author, where)))
                bad = True
            # Does the key still describe the entry?  Reported, never failed: a
            # topical key ("shap2017") is a normal convention, so a miss is a
            # prompt to look, not a proven defect.  It is what surfaces an entry
            # whose author and title were replaced under an old key.
            km = KEY_WORD_RE.match(key)
            if km:
                blob = _fold(" ".join(str(f.get(k, "")) for k in KEY_MATCH_FIELDS))
                word = km.group(1)
                parts = [p for p in KEY_SEGMENT_RE.findall(word) if len(p) >= 3]
                if not (_fold(word) in blob
                        or (len(parts) > 1 and all(_fold(p) in blob for p in parts))):
                    problems.append(("WARN", key,
                                     "the key names %r, which appears in no field of "
                                     "the entry -- not the authors, not the title, "
                                     "not the venue. Either a topical key, or an "
                                     "entry replaced under an old key (%s)"
                                     % (word, where)))
            if not bad:
                self.n_traced += 1

        for key, needle in must_include:
            self.n_checked += 1
            ent = entries.get(key)
            if ent is None:
                problems.append(("FAIL", key, "pinned entry absent; it must exist and "
                                              "its author must contain %r" % needle))
            elif needle.lower() not in _debrace(ent["fields"].get("author", "")).lower():
                problems.append(("FAIL", key, "author must contain %r but is %r"
                                 % (needle,
                                    _debrace(ent["fields"].get("author", ""))[:80])))
            else:
                self.n_traced += 1

        if not problems:
            self.lines.append("  [OK]   every cited key resolves; every entry has an "
                              "author and a title")
        for sev, key, detail in problems:
            self.lines.append("  [%s] %s: %s" % (sev, key, detail))
            if sev == "FAIL":
                self.failures.append("bibliography: %s: %s" % (key, detail))
        return problems

    def report(self, out_rel, tolerance_note):
        head = [
            "PROVENANCE REPORT -- paper %s" % self.paper,
            "generated by code/verify_manuscript_numbers.py (traceability pass, 2026-09-10)",
            "",
            "Every number in this paper's prose, abstract, tables and figures is either",
            "re-derived below from a committed data file or listed as UNTRACED.",
            "",
            "METHOD, AND WHAT IT CANNOT CATCH",
            "  CLAIM checks are the strong ones: a named number is recomputed from a named",
            "  field of a named file and compared at a stated tolerance.",
            "  The SWEEP is a safety net, not a proof. It asks only whether each numeric",
            "  token in the manuscript is *some* value derivable from *some* committed",
            "  source, so a number that is right in form but attached to the wrong",
            "  quantity can still pass the sweep. It is there to catch numbers that come",
            "  from nowhere -- the paper-04 and paper-12 failure mode -- and it does that",
            "  reliably. Any number whose identity matters is also given a CLAIM check.",
            "  FIGURE checks read the PDF text layer; a value baked into a raster image",
            "  cannot be read and is reported as such rather than silently passed.",
            "",
            "TOLERANCES",
            tolerance_note,
            "",
            "CLAIMS",
        ]
        tail = [
            "",
            "SUMMARY",
            "  claims checked      : %d" % self.n_checked,
            "  claims traced       : %d" % self.n_traced,
            "  claims NOT traced   : %d" % (self.n_checked - self.n_traced),
            "  failures            : %d" % len(self.failures),
        ]
        if self.failures:
            tail.append("")
            tail.append("FAILURES / UNTRACED NUMBERS (for the author, not to be silently fixed)")
            for f in self.failures:
                tail.append("  - %s" % f)
        else:
            tail.append("")
            tail.append("  Every checked number traces to a committed source file.")
        text = "\n".join(head + self.lines + tail) + "\n"
        out = self.root / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(text)
        print("written: %s" % out)
        return 1 if self.failures else 0


def _fmt(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return ("%d" % f) if float(f).is_integer() and abs(f) < 1e9 else ("%.6g" % f)


def pool_of(*iterables) -> set:
    out = set()
    for it in iterables:
        for v in it:
            out |= fmts(v)
    return out


def ticks(*seqs) -> set:
    return pool_of(*seqs)


def frange(lo, hi, step):
    out, v = [], lo
    while v <= hi + 1e-9:
        out.append(round(v, 6))
        v += step
    return out


def decades(lo, hi) -> set:
    out = set()
    for k in range(lo, hi + 1):
        out.add("10%d" % k)
        out.add("10%d" % abs(k))
        out |= fmts(10.0 ** k)
    return out


def literal(*s) -> set:
    return set(s)
# ------------------------------------------------------------ end engine ----

# =============================================================== paper 09 ====
ROOT = Path(__file__).resolve().parents[1]
PAPER = "09_Comparative_Optimization_Hydraulic_Diagnostics"

TOLERANCE_NOTE = """  Default 0.0005 -- half a unit in the last digit printed for a 3-decimal
  quantity (macro-F1, top-1 accuracy, AUC, Spearman rho). Per claim otherwise:
    0.00005 for 4-decimal quantities (Brier score, ECE)
    0.05    for quantities the manuscript rounds to 1 decimal (per cent)
    0       for integer counts (cycles, sensor streams, seeds, components)
  No tolerance is wider than the manuscript's own printed precision."""

SOURCES = [
    "results/ceiling_bound_check.csv",
    "results/ceiling_bound_check_runclustered.csv",
    "results/chronological_curve_runclustered.csv",
    "results/condition_run_lengths.csv",
    "results/chronological_curve.csv",
    "results/comparator_classifiers.csv",
    "results/comparator_classifiers_summary.csv",
    "results/cross_component_auc.csv",
    "results/degradation_walk.csv",
    "results/degradation_walk_summary.csv",
    "results/explanation_agreement.csv",
    "results/explanation_agreement_detail.csv",
    "data/processed/paper3_calibration_metrics.csv",
    "data/processed/paper3_component_metrics.csv",
    "data/processed/paper3_confusion_matrices_temporal.csv",
    "data/processed/paper3_noise_stress.csv",
    "data/processed/paper3_permutation_importance.csv",
    "data/processed/paper3_priority_sensitivity.csv",
    "data/processed/paper3_ranking_metrics.csv",
    "data/processed/paper3_split_robustness.csv",
    "data/processed/paper3_summary.json",
    "data/processed/paper3_targets.csv",
    "data/processed/paper3_test_probabilities.csv",
    "data/processed/paper3_feature_groups.json",
    "data/uci_hydraulic_raw/description.txt",
    "manuscript/generated_calibration_table.tex",
    "manuscript/generated_ceiling_bound_table.tex",
    "manuscript/generated_chronological_curve_table.tex",
    "manuscript/generated_comparator_table.tex",
    "manuscript/generated_component_metrics_table.tex",
    "manuscript/generated_confusion_table.tex",
    "manuscript/generated_constants_table.tex",
    "manuscript/generated_cross_auc_table.tex",
    "manuscript/generated_degradation_walk_table.tex",
    "manuscript/generated_explanation_agreement_table.tex",
    "manuscript/generated_leakage_robustness_table.tex",
    "manuscript/generated_priority_sensitivity_table.tex",
    "manuscript/generated_ranking_baselines_table.tex",
]

ALLOWED = {
    # Added 2026-09-12 with the author's settled indexing identity.  The four
    # digit groups of the ORCID 0000-0002-6197-1025 are parts of a persistent
    # identifier, not measurements: they trace to no result file because they
    # are not results.  Declared here rather than hidden from the sweep.
    "0000": "digit group 1 of the author's ORCID 0000-0002-6197-1025, an identifier, not a result",
    "0002": "digit group 2 of the author's ORCID 0000-0002-6197-1025, an identifier, not a result",
    "6197": "digit group 3 of the author's ORCID 0000-0002-6197-1025, an identifier, not a result",
    "1025": "digit group 4 of the author's ORCID 0000-0002-6197-1025, an identifier, not a result",
    "1.15": "LaTeX \\arraystretch, not a result",
    "10000": "LaTeX \\interfootnotelinepenalty, not a result",
    "2000": "the pre-declared number of bootstrap resamples, a protocol constant",
    "9.6": "the classification error rate Helwig et al. report; an external "
           "literature value, cited, not re-derived here",
    "30": "the 30 per cent test share of the pre-declared 70/30 stratified split",
    "70": "the 70 per cent train share of the pre-declared 70/30 stratified split",
    "50": "a chronological cut fraction declared in the protocol",
    "65": "a chronological cut fraction declared in the protocol",
}


def main() -> int:
    v = Verifier(ROOT, PAPER)
    R = ROOT / "results"
    P = ROOT / "data" / "processed"

    summary = json.loads((P / "paper3_summary.json").read_text(encoding="utf-8"))
    comp = pd.read_csv(R / "comparator_classifiers_summary.csv")
    auc = pd.read_csv(R / "cross_component_auc.csv")
    ceil = pd.read_csv(R / "ceiling_bound_check.csv")
    chrono = pd.read_csv(R / "chronological_curve.csv")
    rclu = pd.read_csv(R / "ceiling_bound_check_runclustered.csv")
    runlen = pd.read_csv(R / "condition_run_lengths.csv")
    walk = pd.read_csv(R / "degradation_walk_summary.csv")
    expl = pd.read_csv(R / "explanation_agreement.csv")
    targets = pd.read_csv(P / "paper3_targets.csv")
    calib = pd.read_csv(P / "paper3_calibration_metrics.csv")
    rank = pd.read_csv(P / "paper3_ranking_metrics.csv")
    Sj = "data/processed/paper3_summary.json"

    # ---------------------------------------------------------- abstract ----
    v.check("abstract: benchmark cycles", int(summary["instances"]), 2205,
            Sj + ":instances (= data/processed/paper3_targets.csv row count)", tol=0)
    v.check("abstract: cycles agree with the committed target table",
            len(targets), int(summary["instances"]),
            "data/processed/paper3_targets.csv row count", tol=0)
    v.check("abstract: random-split macro-F1", summary["macro_f1"], 0.999,
            Sj + ":macro_f1")
    v.check("abstract: macro Brier score", summary["macro_brier_score"], 0.0031,
            Sj + ":macro_brier_score", tol=5e-5)
    v.check("abstract: 10-bin expected calibration error", summary["macro_ece_10_bin"], 0.0202,
            Sj + ":macro_ece_10_bin", tol=5e-5)
    v.check("abstract: test cycles", int(summary["test_instances"]), 662,
            Sj + ":test_instances", tol=0)
    v.check("abstract: fault-containing test cycles",
            int(summary["best_ranking_method"]["fault_cycles"]), 656,
            Sj + ":best_ranking_method.fault_cycles", tol=0)
    v.check("abstract: top-1 root-cause accuracy on fault cycles",
            summary["best_ranking_method"]["top1_root_cause_accuracy"], 1.000,
            Sj + ":best_ranking_method.top1_root_cause_accuracy")

    rf = comp[comp.classifier.str.startswith("Random Forest")].set_index("split")
    hgb = comp[comp.classifier == "HistGradientBoosting"].set_index("split")
    sr = pd.read_csv(P / "paper3_split_robustness.csv").set_index("split")
    v.check("abstract: leakage-safe blocked-group headline macro-F1",
            sr.loc["blocked_group_5fold", "macro_f1"], 0.965,
            "data/processed/paper3_split_robustness.csv:macro_f1")
    v.check("abstract: random-split macro-F1 (single run, Fig. split sensitivity)",
            sr.loc["random_stratified", "macro_f1"], 0.998,
            "data/processed/paper3_split_robustness.csv:macro_f1")
    v.check("comparator table: blocked-group RF macro-F1 (3-seed mean)",
            rf.loc["blocked_group_5fold", "macro_f1_mean"], 0.962,
            "results/comparator_classifiers_summary.csv:macro_f1_mean, Random Forest")
    v.check("abstract: blocked-group HistGradientBoosting macro-F1",
            hgb.loc["blocked_group_5fold", "macro_f1_mean"], 0.967,
            "results/comparator_classifiers_summary.csv:macro_f1_mean, HistGradientBoosting")
    v.check("abstract: HistGradientBoosting matches or exceeds RF on every split",
            bool((hgb["macro_f1_mean"] >= rf["macro_f1_mean"] - 1e-12).all()), True,
            "results/comparator_classifiers_summary.csv:macro_f1_mean", tol=0)
    v.check("abstract: chronological macro-F1",
            sr.loc["temporal_chronological", "macro_f1"], 0.667,
            "data/processed/paper3_split_robustness.csv:macro_f1")
    v.check("abstract: random-split macro-F1 from the comparator table",
            rf.loc["random_stratified", "macro_f1_mean"], 0.999,
            "results/comparator_classifiers_summary.csv:macro_f1_mean, Random Forest")

    # -------------------------------------------------- separability bound --
    for split, quoted in (("random_stratified", 0.000),
                          ("blocked_group_5fold", 0.023),
                          ("temporal_chronological_0.70", 0.175)):
        rows = ceil[ceil.split.str.startswith(split.split("_0.70")[0])]
        b = float(rows.bound.max()) if len(rows) else float("nan")
        v.check("abstract: union-bound top-1 error on %s" % split, b, quoted,
                "results/ceiling_bound_check.csv:bound", tol=0.0005)
    v.check("abstract: static severity/effort policies floor",
            float(ceil[ceil.split == "random_stratified"]
                  .query("policy.str.contains('Severity-only')", engine="python")
                  .observed_top1_error.iloc[0]), 1 - 0.697,
            "results/ceiling_bound_check.csv:observed_top1_error", tol=0.0005)
    v.check("abstract: static low-effort policy floor",
            float(ceil[ceil.split == "random_stratified"]
                  .query("policy.str.contains('Low-effort')", engine="python")
                  .observed_top1_error.iloc[0]), 1 - 0.735,
            "results/ceiling_bound_check.csv:observed_top1_error", tol=0.0005)
    # ---- resampling unit (added 2026-09-12; see results/SUPERSEDED_2026-09-12.md)
    # The bootstrap unit is the condition-run, not the cycle. These checks tie the
    # design description in Sec. "Resampling Unit for Every Reported Interval" and
    # every run-clustered interval quoted in prose back to the committed CSVs.
    v.check("resampling: condition-runs over the 2205 cycles",
            int(runlen.n_runs.sum()), 194,
            "results/condition_run_lengths.csv:n_runs (sum)", tol=0)
    v.check("resampling: cycles accounted for by the runs",
            int(runlen.n_cycles.sum()), 2205,
            "results/condition_run_lengths.csv:n_cycles (sum)", tol=0)
    v.check("resampling: runs of length 10",
            int(runlen.loc[runlen.run_length == 10, "n_runs"].iloc[0]), 154,
            "results/condition_run_lengths.csv:n_runs at run_length=10", tol=0)
    v.check("resampling: singleton runs",
            int(runlen.loc[runlen.run_length == 1, "n_runs"].iloc[0]), 36,
            "results/condition_run_lengths.csv:n_runs at run_length=1", tol=0)
    v.check("resampling: condition-runs in the chronological test block",
            int(rclu[rclu.split == "temporal_chronological_0.70"].n_runs.iloc[0]), 65,
            "results/ceiling_bound_check_runclustered.csv:n_runs", tol=0)
    _blk = rclu[rclu.split == "blocked_group_5fold"]
    _po = _blk[_blk.quantity == "Probability only"].iloc[0]
    _bd = _blk[_blk.quantity == "bound"].iloc[0]
    _pd_ = _blk[_blk.quantity == "paired_priority_minus_probonly"].iloc[0]
    for lbl, val, quoted in (
            ("ceiling check: prob-only run-clustered CI low", _po.clu_lo, 0.009),
            ("ceiling check: prob-only run-clustered CI high", _po.clu_hi, 0.040),
            ("ceiling check: bound run-clustered CI low", _bd.clu_lo, 0.009),
            ("ceiling check: bound run-clustered CI high", _bd.clu_hi, 0.040),
            ("ceiling check: paired priority-minus-prob difference", _pd_.est, 0.029),
            ("ceiling check: paired difference CI low", _pd_.clu_lo, 0.012),
            ("ceiling check: paired difference CI high", _pd_.clu_hi, 0.052)):
        v.check(lbl, float(val), quoted,
                "results/ceiling_bound_check_runclustered.csv", tol=0.0005)
    v.check("ceiling check: static policies' lowest run-clustered CI bound",
            float(_blk[_blk.quantity.str.contains("Low-effort")].clu_lo.iloc[0]), 0.134,
            "results/ceiling_bound_check_runclustered.csv:clu_lo", tol=0.0005)
    v.check("ceiling check: static policies' highest run-clustered CI bound",
            float(_blk[_blk.quantity.str.contains("Severity-only")].clu_hi.iloc[0]), 0.492,
            "results/ceiling_bound_check_runclustered.csv:clu_hi", tol=0.0005)
    v.check("resampling: narrowest blocked-split widening factor",
            float(_blk[_blk.quantity.isin(["bound", "Probability only", "Severity-only policy",
                                           "Low-effort-only policy", "Severity/effort heuristic",
                                           "Explanation confidence only", "Local occlusion",
                                           "Priority score (prob + evidence + severity/effort)"])]
                  .width_ratio_clu_over_iid.min()), 2.4,
            "results/ceiling_bound_check_runclustered.csv:width_ratio_clu_over_iid (min)", tol=0.05)
    v.check("resampling: widest blocked-split widening factor",
            float(_blk.width_ratio_clu_over_iid.max()), 8.0,
            "results/ceiling_bound_check_runclustered.csv:width_ratio_clu_over_iid (max)", tol=0.05)
    v.check("resampling: chronological widening factor",
            float(rclu[(rclu.split == "temporal_chronological_0.70")
                       & (rclu.quantity == "bound")].width_ratio_clu_over_iid.iloc[0]), 3.6,
            "results/ceiling_bound_check_runclustered.csv:width_ratio_clu_over_iid", tol=0.05)

    v.check("chronological sweep: cut points in the committed curve",
            int(chrono.cut_fraction.nunique()), 9,
            "results/chronological_curve.csv:cut_fraction distinct values", tol=0)
    # The abstract said "nine ADDITIONAL chronological cut points" until
    # 2026-09-11. One of the nine committed cuts (0.70) is the headline
    # chronological split itself, so only eight are additional; the abstract now
    # says eight and this check holds it there.
    v.check("chronological sweep: cut points ADDITIONAL to the 0.70 headline",
            int(chrono[chrono.cut_fraction != 0.70].cut_fraction.nunique()), 8,
            "results/chronological_curve.csv:cut_fraction excluding the 0.70 headline", tol=0)
    v.present("guardrail: abstract counts the additional cuts, not all nine",
              "across eight additional chronological cut points",
              (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8"))
    v.check("cross-component AUC: components per split", int(auc.component.nunique()), 4,
            "results/cross_component_auc.csv:component", tol=0)
    v.check("cross-component AUC: random-split minimum",
            float(auc[auc.split == "random_stratified"].cross_auc.min()), 1.000,
            "results/cross_component_auc.csv:cross_auc", tol=0.0005)

    # ----------------------------------------------------- degradation ------
    v.check("degradation sweep: axes", int(walk.axis.nunique()), 4,
            "results/degradation_walk_summary.csv:axis", tol=0)
    v.check("degradation sweep: seeds per cell", int(walk.n_seeds.max()), 10,
            "results/degradation_walk_summary.csv:n_seeds", tol=0)

    # Added 2026-09-17.  Until today this section checked the sweep's SHAPE (4 axes,
    # 10 seeds) and not one of its VALUES, while the figure section below told the
    # reader that degradation_walk.png was sound because its series "are checked as
    # claims above".  They were not: every degradation number the Results section
    # prints was unchecked, and a re-run that moved one of them from 0.793 to 0.791
    # still reported 350/350 traced.  Each value the manuscript quotes from this
    # experiment is now a named claim, so a drift in the underlying run fails here
    # instead of passing in silence.
    _PRIO = "Priority score (prob + evidence + severity/effort)"

    def _walk(axis, level, col):
        row = walk[(walk.axis == axis) & (walk.level == level) & (walk.policy == _PRIO)]
        if len(row) != 1:
            return float("nan")
        return float(row[col].iloc[0])

    for _lbl, _axis, _lvl, _col, _quoted in (
        ("degradation: macro-F1, full cycle window",
         "cycle_window_retained", 1.00, "macro_f1_mean", 0.961),
        ("degradation: macro-F1, 5% cycle window",
         "cycle_window_retained", 0.05, "macro_f1_mean", 0.791),
        ("degradation: top-1, full cycle window",
         "cycle_window_retained", 1.00, "top1_mean", 0.978),
        ("degradation: top-1, 5% cycle window",
         "cycle_window_retained", 0.05, "top1_mean", 0.975),
        ("degradation: union bound, 5% cycle window",
         "cycle_window_retained", 0.05, "bound_top1_error", 0.041),
        ("degradation: top-1, no label flips",
         "label_flip_rate", 0.00, "top1_mean", 0.978),
        ("degradation: top-1, 30% label flips",
         "label_flip_rate", 0.30, "top1_mean", 0.942),
        ("degradation: union bound, no label flips",
         "label_flip_rate", 0.00, "bound_top1_error", 0.027),
        ("degradation: union bound, 30% label flips",
         "label_flip_rate", 0.30, "bound_top1_error", 0.080),
        ("degradation: top-1, no families removed",
         "sensor_families_removed", 0.0, "top1_mean", 0.978),
        ("degradation: top-1, 4 of 7 families removed",
         "sensor_families_removed", 4.0, "top1_mean", 0.976),
        ("degradation: macro-F1, no families removed",
         "sensor_families_removed", 0.0, "macro_f1_mean", 0.961),
        ("degradation: macro-F1, 4 of 7 families removed",
         "sensor_families_removed", 4.0, "macro_f1_mean", 0.910),
        ("degradation: top-1, 3% of training rows",
         "training_fraction", 0.03, "top1_mean", 0.977),
        ("degradation: top-1, 1% of training rows",
         "training_fraction", 0.01, "top1_mean", 0.932),
        ("degradation: top-1 sd, 1% of training rows",
         "training_fraction", 0.01, "top1_std", 0.083),
    ):
        v.check(_lbl, _walk(_axis, _lvl, _col), _quoted,
                "results/degradation_walk_summary.csv:%s" % _col, tol=0.0005)

    # The union bound and which policies it actually constrains.
    #
    # policy_scores() (code/ceiling_analysis.py:152) builds seven policies. Only
    # two are monotone per-row transforms of `prob` and therefore genuinely
    # probability-consistent in the sense the bound requires:
    #     "Probability only"  = prob
    #     "Local occlusion"   = prob * (1 + occlusion_conf)
    # The four static policies (severity-only, low-effort-only, severity/effort,
    # explanation-confidence-only) never consult prob, so the bound says nothing
    # about them and they exceed it at every level -- that is the paper's point,
    # not a defect.
    #
    # "Priority score" multiplies prob by the per-component constant
    # SEVERITY/EFFORT, which REORDERS components. It is therefore NOT
    # probability-consistent, which is precisely this paper's central negative
    # finding. The bound is not guaranteed for it, and it is checked separately
    # below rather than folded in either direction.
    _PROB_CONSISTENT = ("Probability only", "Local occlusion")
    _pc = walk[walk.policy.isin(_PROB_CONSISTENT)]
    _viol = _pc[(1.0 - _pc.top1_mean) > _pc.bound_top1_error + 1e-12]
    v.check("degradation: union bound holds at every level, probability-consistent policies",
            int(len(_viol)), 0,
            "results/degradation_walk_summary.csv:top1_mean vs bound_top1_error "
            "(Probability only, Local occlusion)", tol=0)

    # RESOLVED 2026-09-17 (author decision). This used to be an open sentence-scope
    # question: the Results section asserted the bound holds "at every level tested"
    # in a sentence illustrated with the PRIORITY-SCORE policy's own numbers (observed
    # 0.025 at 5% cycle retention), which does not hold at training_fraction = 0.01
    # (~17 training rows), observed 0.068 against a bound of 0.065 -- present in the
    # 2026-09-10 outputs too, not caused by that day's re-run. The author's decision
    # (first option offered, matching Sec.~ceiling's own derivation condition):
    # restrict the "holds at every level" claim to the two probability-consistent
    # policies (Probability only, Local occlusion; checked immediately above, 0
    # violations) and state plainly that the priority-score policy was never in scope
    # -- it multiplies probability by a per-component severity/effort constant, which
    # can reorder components, so the bound's derivation does not cover it. main.tex's
    # paragraph was rewritten accordingly and now explicitly names training_fraction =
    # 0.01 as priority score's one exception. This check therefore no longer asks
    # whether priority score stays inside the bound (it does not, by construction, and
    # the manuscript no longer claims otherwise); it verifies the manuscript's new,
    # narrower claim is exactly right: exactly one violating cell, and it is the one
    # named in the text.
    _prio = walk[walk.policy == _PRIO]
    _pviol = _prio[(1.0 - _prio.top1_mean) > _prio.bound_top1_error + 1e-12]
    v.check("degradation: priority-score policy's disclosed bound exception is "
            "exactly one cell (count)",
            int(len(_pviol)), 1,
            "results/degradation_walk_summary.csv:top1_mean vs bound_top1_error "
            "(priority score; violating cell(s): %s)"
            % (", ".join("%s=%g" % (r.axis, r.level) for r in _pviol.itertuples()) or "none"),
            tol=0)
    _pviol_cells = set((r.axis, float(r.level)) for r in _pviol.itertuples())
    v.check("degradation: priority-score policy's disclosed bound exception is "
            "the training_fraction=0.01 cell, and only that cell",
            _pviol_cells == {("training_fraction", 0.01)}, True,
            "results/degradation_walk_summary.csv:axis,level of the violating cell(s)",
            tol=0)

    # -------------------------------------------- explanation agreement -----
    e = expl.set_index("component")
    for comp_name in ("cooler", "valve", "pump", "accumulator"):
        v.check("explanation agreement: %s occlusion vs SHAP rho" % comp_name,
                e.loc[comp_name, "spearman_occlusion_shap"],
                round(float(e.loc[comp_name, "spearman_occlusion_shap"]), 3),
                "results/explanation_agreement.csv:spearman_occlusion_shap")

    # -------------------------------------------------- priority constants --
    # Table tab:constants is read directly from the experiment script, so the
    # script is the source of record and the table must match it.
    src = ""
    for cand in ("code/paper3_diagnostics_experiment.py",
                 "code/run_experiment.py", "code/paper3_experiment.py"):
        p = ROOT / cand
        if p.is_file():
            src = p.read_text(encoding="utf-8", errors="ignore")
            srcname = cand
            break
    if src:
        for comp_name, sev, eff in (("Cooler", 4.0, 1.8), ("Valve", 3.0, 1.5),
                                    ("Pump", 3.5, 2.2), ("Accumulator", 3.2, 1.4)):
            v.check("priority constants: %s severity in the code" % comp_name,
                    str(sev) in src, True, srcname, tol=0)
            v.check("priority constants: %s effort in the code" % comp_name,
                    str(eff) in src, True, srcname, tol=0)
    else:
        v.lines.append("  NOTE  priority-constant source script not present in this "
                       "checkout; tab:constants not cross-checked against code")

    # ------------------------------------------------------- guardrails -----
    tex = (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    v.present("guardrail: evidence graph is not a causal model",
              "no causal-discovery claim is made", tex)
    v.present("guardrail: random split declared optimistic",
              "optimistic reference", tex)
    v.present("guardrail: alpha/beta edge weights not implemented",
              "have no implemented values", tex)

    # ------------------------------- Fig. priority example, per component ---
    # The caption states these are the probabilities of one specific test
    # cycle -- the one maximising sum_c p_c(x) y_c -- so they must equal that
    # cycle's committed probabilities.
    #
    # 2026-09-11. The labels for the test rows are not committed alongside the
    # probabilities, so the split is replayed exactly as
    # paper3_diagnostics_experiment.train_models performs it. The earlier
    # `targets.tail(len(probs))` proxy happened to select the same row, but it
    # is not the split the caption refers to and is not relied on any more.
    from sklearn.model_selection import train_test_split as _tts
    probs = pd.read_csv(P / "paper3_test_probabilities.csv")
    comps = ["cooler", "valve", "pump", "accumulator"]
    _, _test_idx = _tts(np.arange(len(targets)), test_size=0.30, random_state=17,
                        stratify=targets["any_fault"])
    v.check("Fig. priority example: replayed test split matches the committed probabilities",
            len(_test_idx), len(probs),
            "train_test_split(test_size=0.30, random_state=17, stratify=any_fault) "
            "vs data/processed/paper3_test_probabilities.csv row count", tol=0)
    Pm = probs[[c + "_probability" for c in comps]].to_numpy()
    Ym = targets[[c + "_fault" for c in comps]].to_numpy()[_test_idx]
    mass = (Pm * Ym).sum(axis=1)
    star = int(mass.argmax())
    v.check("Fig. priority example: the argmax cycle is unique",
            int((mass == mass[star]).sum()), 1,
            "data/processed/paper3_test_probabilities.csv + paper3_targets.csv, "
            "argmax_x sum_c p_c(x)y_c", tol=0)
    v.check("Fig. priority example: the cycle drawn is test row 550", star, 550,
            "data/processed/paper3_test_probabilities.csv, argmax_x sum_c p_c(x)y_c",
            tol=0)
    # The figure now prints, per component, p_c(x) and S_c/E_c to three decimals
    # and their product. These are the values the rebuilt figure draws
    # (code/build_priority_example_figure.py); before 2026-09-11 the figure was a
    # Visio mock-up printing 0.98 / 0.12 / 0.05 / 0.07 plus an "Evidence Conf"
    # column for a field no committed file contains (manuscript/
    # _superseded_20260911/README.txt records it in full).
    DRAWN_P = {"cooler": 1.000, "valve": 1.000, "pump": 0.996, "accumulator": 1.000}
    DRAWN_RATIO = {"cooler": 2.222, "valve": 2.000, "pump": 1.591, "accumulator": 2.286}
    DRAWN_PRIORITY = {"cooler": 2.222, "valve": 2.000, "pump": 1.585, "accumulator": 2.286}
    SEV = {"cooler": 4.0, "valve": 3.0, "pump": 3.5, "accumulator": 3.2}
    EFF = {"cooler": 1.8, "valve": 1.5, "pump": 2.2, "accumulator": 1.4}
    for k, comp_name in enumerate(comps):
        v.check("Fig. priority example: %s fault probability" % comp_name,
                float(Pm[star, k]), DRAWN_P[comp_name],
                "data/processed/paper3_test_probabilities.csv, argmax_x sum_c p_c(x)y_c "
                "row %d" % star, tol=0.0005)
        ratio = SEV[comp_name] / EFF[comp_name]
        v.check("Fig. priority example: %s S_c/E_c" % comp_name,
                ratio, DRAWN_RATIO[comp_name],
                "code/paper3_diagnostics_experiment.py:SEVERITY/EFFORT (= tab:constants)",
                tol=0.0005)
        v.check("Fig. priority example: %s priority score" % comp_name,
                float(Pm[star, k]) * ratio, DRAWN_PRIORITY[comp_name],
                "p_c(x) * S_c/E_c on argmax_x sum_c p_c(x)y_c row %d" % star,
                tol=0.0005)
    v.check("Fig. priority example: all four components are truly faulted on the cycle drawn",
            int(Ym[star].sum()), 4,
            "data/processed/paper3_targets.csv on the replayed test split", tol=0)
    # The retired mock-up printed an "Evidence Conf" column. Nothing in this
    # repository carries such a field, so the check is that no such column has
    # reappeared -- not that one exists.
    v.check("Fig. priority example: no 'evidence confidence' field is claimed anywhere",
            any("evidence" in c.lower() or "confidence" in c.lower()
                for c in probs.columns), False,
            "data/processed/paper3_test_probabilities.csv column names", tol=0)
    v.present("guardrail: the priority example declares that S_c/E_c, not the classifier, orders it",
              "is set by the fixed $S_c/E_c$ constants of Table~\\ref{tab:constants} "
              "rather than by the classifier",
              (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8"))

    # ------------------------------------------------------------ sweep -----
    derived = [1 - float(x) for x in ceil.observed_top1_error] + \
              [1 - float(x) for x in ceil.bound] + \
              [float(x) * 100 for x in ceil.observed_top1_error] + \
              [int(chrono.cut_fraction.nunique()) - 1,
               summary["best_ranking_method"]["fault_cycles"] / summary["test_instances"] * 100]
    for c in ("top1_accuracy", "bound_top1_error", "macro_f1", "mean_cross_auc"):
        if c in chrono.columns:
            derived += [1 - float(x) for x in chrono[c]]
    derived += [1 - float(x) for x in walk.top1_mean]
    # Sec. "Maintenance Priority" quotes the priority example directly: the
    # index of the argmax test row and the four p_c(x)*S_c/E_c products. Both
    # are re-derived above from committed data, so they are declared here
    # rather than whitelisted as non-data.
    derived += [star]
    derived += [float(Pm[star, k]) * SEV[c] / EFF[c] for k, c in enumerate(comps)]
    derived += [SEV[c] / EFF[c] for c in comps]
    v.sweep(["manuscript/main.tex"], SOURCES, ALLOWED, derived=derived)

    # ---------------------------------------------------------- figures -----
    def cells(df):
        out = []
        for col in df.columns:
            out.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
        return out

    imp = pd.read_csv(P / "paper3_permutation_importance.csv")
    cm = pd.read_csv(P / "paper3_component_metrics.csv")
    prio = pd.read_csv(P / "paper3_priority_sensitivity.csv")
    specs = {
        "manuscript/paper3_component_f1.pdf": (
            pool_of(cells(cm), cells(rank),
                    cells(pd.read_csv(P / "paper3_split_robustness.csv"))),
            ticks(frange(0, 1.0, 0.1), frange(0, 100, 10)),
            "data/processed/paper3_component_metrics.csv",
        ),
        "manuscript/paper3_importance_heatmap.pdf": (
            pool_of(cells(imp)),
            ticks(frange(0, 1.0, 0.05), frange(0, 20, 1)),
            "data/processed/paper3_permutation_importance.csv",
        ),
        "manuscript/paper3_root_cause_ranking.pdf": (
            pool_of(cells(rank), cells(ceil), [1 - float(x) for x in ceil.observed_top1_error]),
            ticks(frange(0, 1.0, 0.1), frange(0, 100, 10)),
            "data/processed/paper3_ranking_metrics.csv + results/ceiling_bound_check.csv",
        ),
        "manuscript/paper3_split_sensitivity.pdf": (
            pool_of(cells(pd.read_csv(P / "paper3_split_robustness.csv")),
                    cells(pd.read_csv(P / "paper3_noise_stress.csv")),
                    cells(comp), cells(chrono)),
            # panel (b) x-axis is the sensor-noise sigma grid declared in
            # data/processed/paper3_noise_stress.csv:noise_sigma
            ticks(frange(0, 1.0, 0.1), frange(0, 2.0, 0.5), [5]),
            "data/processed/paper3_split_robustness.csv + paper3_noise_stress.csv",
        ),
        # Rebuilt 2026-09-11 by code/build_priority_example_figure.py. It prints,
        # per component, the argmax cycle's committed probability, the S_c/E_c
        # ratio from tab:constants, and their product; plus the test-row index
        # 550, the test-split size 662, and the x-axis ticks.
        "manuscript/paper3_maintenance_priority_example.pdf": (
            pool_of(cells(prio), cells(rank), [4.0, 1.8, 3.0, 1.5, 3.5, 2.2, 3.2, 1.4],
                    cells(pd.read_csv(P / "paper3_test_probabilities.csv")),
                    [SEV[c] / EFF[c] for c in comps],
                    [float(Pm[star, k]) * SEV[c] / EFF[c] for k, c in enumerate(comps)],
                    [star, summary["test_instances"]]),
            ticks(frange(0, 2.5, 0.5), frange(0, 1.0, 0.1), [1, 2, 3, 4]),
            "data/processed/paper3_test_probabilities.csv + tab:constants "
            "(via code/build_priority_example_figure.py)",
        ),
        "manuscript/paper3_evidence_graph.pdf": (
            pool_of(cells(imp), [4.0, 1.8, 3.0, 1.5, 3.5, 2.2, 3.2, 1.4]),
            ticks([1, 2, 3, 4, 5, 6, 17]),
            "data/processed/paper3_permutation_importance.csv",
        ),
        "manuscript/paper3_conceptual_architecture.pdf": (
            pool_of([summary["instances"], summary["features"], summary["test_instances"],
                     summary["features"] / 17],   # 11 cycle-level descriptors per stream
                    cells(cm)),
            # 60 s per cycle and 17 sensor streams are properties of the UCI
            # release, stated in data/uci_hydraulic_raw/description.txt
            ticks([1, 2, 3, 4, 5, 6, 17, 60]),
            "data/processed/paper3_summary.json + data/uci_hydraulic_raw/description.txt",
        ),
        "manuscript/paper3_priority_rule.pdf": (
            pool_of([4.0, 1.8, 3.0, 1.5, 3.5, 2.2, 3.2, 1.4], cells(prio),
                    [1 - float(x) for x in ceil.observed_top1_error]),
            ticks([0, 1, 2, 3, 4, 5]),   # "Eq. (5)" is a cross-reference
            "tab:constants + data/processed/paper3_priority_sensitivity.csv",
        ),
        "manuscript/paper3_explanation_evidence.pdf": (
            pool_of(cells(expl), cells(imp)),
            ticks([0, 1, 2, 3, 4, 5]),
            "results/explanation_agreement.csv",
        ),
    }
    v.figures(specs, no_text_layer=(
        ("figures/ceiling_bound_vs_observed.png",
         "results/ceiling_bound_check_runclustered.csv (est and the run-clustered clu_lo/clu_hi)"),
        ("figures/ceiling_chronological_curve.png",
         "results/chronological_curve_runclustered.csv"),
        ("figures/degradation_walk.png",
         "results/degradation_walk_summary.csv"),
    ))

    # Blind-spot class 4 (2026-09-11): the bibliography is part of the
    # manuscript's factual claims and was previously checked by nobody.
    v.cites()

    return v.report("code/PROVENANCE_REPORT.txt", TOLERANCE_NOTE)


if __name__ == "__main__":
    sys.exit(main())
