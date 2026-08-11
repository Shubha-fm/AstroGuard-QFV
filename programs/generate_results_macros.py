#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
summary_path = ROOT / "output" / "results" / "final_summary.json"
out_path = ROOT / "output" / "results" / "results_macros.tex"

if not summary_path.exists():
    raise SystemExit(
        f"Missing {summary_path}. Run the experiment before generating manuscript macros."
    )

s = json.loads(summary_path.read_text())

def pct(x):
    return f"{100.0 * float(x):.2f}"

def raw(x, digits=4):
    return f"{float(x):.{digits}f}"

d = s["dataset"]
e = s["ensemble"]
r = s["runtime_assurance"]
f = s.get("formal_verification", {})

lines = [
    rf"\newcommand{{\ExecNTotal}}{{{d['total']}}}",
    rf"\newcommand{{\ExecNTrain}}{{{d['train']}}}",
    rf"\newcommand{{\ExecNVal}}{{{d['validation']}}}",
    rf"\newcommand{{\ExecNTest}}{{{d['test']}}}",
    rf"\newcommand{{\ExecAcc}}{{{pct(e['accuracy'])}}}",
    rf"\newcommand{{\ExecMacroP}}{{{pct(e['macro_precision'])}}}",
    rf"\newcommand{{\ExecMacroR}}{{{pct(e['macro_recall'])}}}",
    rf"\newcommand{{\ExecMacroF}}{{{pct(e['macro_f1'])}}}",
    rf"\newcommand{{\ExecAUROC}}{{{raw(e['macro_auroc'],3)}}}",
    rf"\newcommand{{\ExecECE}}{{{raw(e['ece'],4)}}}",
    rf"\newcommand{{\ExecCoverage}}{{{pct(r['coverage'])}}}",
    rf"\newcommand{{\ExecReleaseAcc}}{{{pct(r['released_accuracy'])}}}",
    rf"\newcommand{{\ExecUnsafeRate}}{{{pct(r['unsafe_release_rate'])}}}",
]
out_path.write_text("\n".join(lines) + "\n")
print(out_path)
