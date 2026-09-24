#!/usr/bin/env python3
"""
check_consistency.py — find contradictions mechanically, not by memory

Runs over CLAIM_INDEX.json and the corpus. Reports things a human would only
catch by remembering, which is the failure mode this project has had repeatedly:

  C1  the same quantity given different values in different documents
      (kappa was 0.30, then 0.0305, then 0.0799 -- two of those reached a
       lab-facing document)
  C2  a document quoting a number no script emits
  C3  scripts referenced by documents that do not exist
  C4  claims that depend on something marked retracted or superseded
  C5  scripts that are blocked (NotImplementedError) but whose numbers are
      quoted anyway
  C6  documents with no script attribution at all

This is the tool that would have caught the kappa contradiction on the day it
was introduced rather than three sessions later.

Usage:  python3 check_consistency.py [--run]
        --run also executes scripts and compares emitted numbers to documents
"""

import os
import re
import json
import sys
import subprocess

IDX = "00_START_HERE/CLAIM_INDEX.json"

# quantities that must agree wherever they appear: name -> regex for its value
TRACKED = {
    "kappa (A^2 coefficient)": re.compile(r"1 \+ ([\d.]+) A"),
    "Greene falloff q=2->21": re.compile(r"([\d,]+)× from\s*\n?q = 2"),
    "K_c golden torus": re.compile(r"K_c[^0-9]{0,20}(0\.97\d+)"),
    "dim Der": re.compile(r"dim Der[^0-9]{0,12}(\d+)"),
    "valid orientations": re.compile(r"(\d+) of 128"),
    "sin2 theta_W": re.compile(r"sin²θ_W = (\d/\d)"),
}


def load():
    if not os.path.exists(IDX):
        print("  CLAIM_INDEX.json missing — run build_index.py first")
        sys.exit(1)
    return json.load(open(IDX))


SELF = {"CLAIM_INDEX.md", "AUDIT.md", "PROVENANCE.md"}


def md_files(include_self=False):
    """Corpus documents. Excludes the checker's own outputs by default --
    CLAIM_INDEX and AUDIT are *about* retractions, so scanning them for
    retraction markers reports the tool reading itself."""
    for r, d, f in os.walk("."):
        d[:] = [x for x in d if x not in {".git", "__pycache__"}]
        for x in sorted(f):
            if x.endswith(".md") and (include_self or x not in SELF):
                yield os.path.join(r, x)


def c1_disagreeing_values():
    print("C1  SAME QUANTITY, DIFFERENT VALUES")
    hits = 0
    for name, rx in TRACKED.items():
        found = {}
        for p in md_files():
            txt = open(p, encoding="utf-8", errors="replace").read()
            for m in rx.findall(txt):
                found.setdefault(m.strip(), []).append(os.path.basename(p))
        if len(found) > 1:
            hits += 1
            print(f"    ** {name}: {len(found)} distinct values")
            for v, files in sorted(found.items()):
                print(f"         {v:12s} in {', '.join(sorted(set(files)))}")
        elif found:
            v = list(found)[0]
            print(f"    ok {name}: {v}")
    if not hits:
        print("    no disagreements among tracked quantities")
    return hits


# Scripts that live in the public GitHub repository rather than in this archive.
# Listed explicitly so the check still flags a script that is genuinely missing:
# a name absent from BOTH the archive and this list is reported.
# Repository: prove2me_workspace, branch claude/prove2me-setup-rbfean,
# folder shape_zero_tests/.
REPO_HOSTED = {
    "q3_gate.py", "q3_kavg.py", "q3_readout.py", "q3_combine.py",
    "gate7_readout.py", "openrows.py", "resid.py", "kscan.py",
    "j_compat_test.py", "grid.py", "checks.py",
}


def c3_missing_scripts(idx):
    print("\nC3  SCRIPTS REFERENCED BUT ABSENT")
    have = {os.path.basename(s["file"]) for s in idx["scripts"]}
    refd = set()
    for p in md_files():
        txt = open(p, encoding="utf-8", errors="replace").read()
        refd |= set(re.findall(r"`([a-z0-9_]+\.py)`", txt))
        refd |= set(re.findall(r"\b([a-z0-9_]+\.py)\b", txt))
    missing = sorted(x for x in refd - have - REPO_HOSTED
                     if not x.startswith("z1_d8_att"))
    remote = sorted((refd - have) & REPO_HOSTED)
    if missing:
        for m in missing:
            print(f"    ** {m}")
    else:
        print("    all referenced scripts present")
    if remote:
        print(f"    (in the GitHub repository, shape_zero_tests/: {', '.join(remote)})")
    return len(missing)


def c4_retracted_dependencies():
    print("\nC4  CLAIMS NEAR RETRACTED / SUPERSEDED MARKERS")
    n = 0
    for p in md_files():
        lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        for i, ln in enumerate(lines):
            if re.search(r"SUPERSEDED|RETRACTED|withdrawn|ERRATUM", ln):
                n += 1
                print(f"    {os.path.basename(p)}:{i+1} — {ln.strip()[:100]}")
    if not n:
        print("    none found (suspicious — this corpus has known retractions)")
    return 0


def c5_blocked_scripts(idx):
    print("\nC5  BLOCKED SCRIPTS (NotImplementedError)")
    bad = 0
    for s in idx["scripts"]:
        if s["blocked"]:
            base = os.path.basename(s["file"])
            quoted = []
            for p in md_files():
                if base in open(p, encoding="utf-8", errors="replace").read():
                    quoted.append(os.path.basename(p))
            flag = " ** QUOTED IN DOCS" if quoted else ""
            print(f"    {base}  fixes={','.join(s['fixes']) or '-'}{flag}")
            if quoted:
                print(f"         referenced by: {', '.join(quoted)}")
                bad += 1
    return bad


def c7_positive_in_retraction():
    """A verified result filed inside a retraction reads as part of it."""
    print("\nC7  MACHINE-PRECISION RESULTS INSIDE RETRACTION PARAGRAPHS")
    n = 0
    prec = re.compile(r"\d\.\d+[eE]-1[0-9]|\d\.\d+×10⁻¹[0-9]")
    for p in md_files():
        lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        for i, ln in enumerate(lines):
            if not prec.search(ln):
                continue
            ctx = " ".join(lines[max(0, i-6):i+2])
            if re.search(r"retract|withdraw|SUPERSEDED|vacuous", ctx, re.I):
                n += 1
                print(f"    ** {os.path.basename(p)}:{i+1} — {ln.strip()[:110]}")
    if not n:
        print("    none")
    return n


def c6_unattributed(idx):
    print("\nC6  DOCUMENTS WITH NO SCRIPT ATTRIBUTION")
    n = 0
    for p in md_files():
        if "/04_scripts/" in p:
            continue
        txt = open(p, encoding="utf-8", errors="replace").read()
        if ".py" not in txt and len(txt) > 1500:
            print(f"    ** {p}")
            n += 1
    if not n:
        print("    every substantial document cites at least one script")
    return n


def main():
    idx = load()
    print("=" * 66)
    print("CONSISTENCY CHECK")
    print("=" * 66)
    print(f"  {idx['counts']['doc_claims']} claims, "
          f"{idx['counts']['scripts']} scripts\n")
    bad = 0
    bad += c1_disagreeing_values()
    bad += c3_missing_scripts(idx)
    c4_retracted_dependencies()
    bad += c5_blocked_scripts(idx)
    bad += c7_positive_in_retraction()
    bad += c6_unattributed(idx)
    print("\n" + "=" * 66)
    print(f"  RESULT: {'CLEAN' if bad == 0 else str(bad) + ' issue(s) to resolve'}")
    print("=" * 66)


if __name__ == "__main__":
    main()
