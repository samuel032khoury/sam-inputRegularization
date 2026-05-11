#!/usr/bin/env python3
"""Compare two confusion.json files (from eval_confusion.py) and print
the numbers we cite in the report's Error Analysis section:

    Out of 10,000 test images, A misses N_A and B misses N_B.
    Of A's misses, B still misses K (overlapping errors), corrects M
    outright, and B *introduces* J new errors that A had right.
    Per-class breakdown table.

Typical use (after running eval_confusion.py for C1 and C6):

    python scripts/error_analysis_compare.py \
        --a /scratch1/$USER/sam_experiments/c1_sgd_baseline/confusion.json \
        --b /scratch1/$USER/sam_experiments/c6_sam_cutmix/confusion.json \
        --out report/error_analysis_numbers.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="confusion.json for run A")
    p.add_argument("--b", required=True, help="confusion.json for run B")
    p.add_argument("--out", default=None,
                   help="optional output text path; otherwise stdout only")
    args = p.parse_args()

    A = json.loads(Path(args.a).read_text())
    B = json.loads(Path(args.b).read_text())

    name_a = Path(A["ckpt_dir"]).name
    name_b = Path(B["ckpt_dir"]).name

    wrong_a = set(A["wrong_indices"])
    wrong_b = set(B["wrong_indices"])

    overlap = wrong_a & wrong_b           # both miss
    a_only = wrong_a - wrong_b            # A misses, B got right
    b_only = wrong_b - wrong_a            # A got right, B misses

    lines = []
    lines.append("=" * 70)
    lines.append(f"Error analysis: {name_a}  vs  {name_b}")
    lines.append("=" * 70)
    lines.append(f"  Test set N           = {A['n']}")
    lines.append(f"  {name_a:30s} errs = {A['errors']}")
    lines.append(f"  {name_b:30s} errs = {B['errors']}")
    lines.append("")
    lines.append(f"  Overlap (both miss)              = {len(overlap)}")
    lines.append(f"  Corrected by B  (A miss, B fix)  = {len(a_only)}")
    lines.append(f"  Introduced by B (A fix, B miss)  = {len(b_only)}")
    lines.append(f"  Net improvement                   = "
                 f"{len(a_only) - len(b_only):+d}")
    lines.append("")
    lines.append("Per-class error counts (A -> B):")
    lines.append(f"  {'class':10s}  {'A_err':>6s}  {'B_err':>6s}  "
                 f"{'delta':>6s}")
    classes = A["class_names"]
    biggest_drop = ("", 0)
    for c in classes:
        ea = A["per_class"][c]["errors"]
        eb = B["per_class"][c]["errors"]
        d = eb - ea
        if (ea - eb) > biggest_drop[1]:
            biggest_drop = (c, ea - eb)
        lines.append(f"  {c:10s}  {ea:6d}  {eb:6d}  {d:+6d}")
    lines.append("")
    if biggest_drop[0]:
        lines.append(
            f"Biggest single-class improvement: {biggest_drop[0]} "
            f"(-{biggest_drop[1]} errors)")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print(text)

    if args.out:
        Path(args.out).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
