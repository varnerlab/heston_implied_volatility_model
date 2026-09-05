"""Copy verified experiment outputs and generate supporting arXiv tables."""
import csv
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "code/results/dynamic_ablation"
SECTIONS = ROOT / "paper-arxiv/sections"
GENERATED = SECTIONS / "generated"
GENERATED.mkdir(exist_ok=True)
for ticker in ("gs", "lly"):
    shutil.copyfile(ROOT / f"code/results/fitted_scenarios/{ticker}_table.tex",
                    GENERATED / f"{ticker}_scenario_table.tex")
shutil.copyfile(RESULTS / "dynamic_ablation_table.tex", GENERATED / "dynamic_ablation_table.tex")
shutil.copyfile(RESULTS / "dynamic_ablation.pdf", SECTIONS / "figures/dynamic_ablation.pdf")

def read_rows(name):
    with (RESULTS / name).open() as stream:
        return list(csv.DictReader(stream))

def write_table(name, columns, header, rows):
    with (GENERATED / name).open("w") as stream:
        stream.write(r"\begin{tabular}{" + columns + "}\n\\toprule\n")
        stream.write(header + r" \\" + "\n\\midrule\n")
        for row in rows:
            stream.write(" & ".join(row) + r" \\" + "\n")
        stream.write("\\bottomrule\n\\end{tabular}\n")

dates = defaultdict(lambda: {"captures": set(), "sessions": set(), "n": 0})
for row in read_rows("corpus_manifest.csv"):
    label = row["path"].split("/")[3].removeprefix("options-")
    dates[label]["captures"].add(row["capture"][:10])
    dates[label]["sessions"].add(row["session"])
    dates[label]["n"] += int(row["filtered_rows"])
write_table("corpus_dates.tex", "lllr", "Snapshot label & Capture date & Underlying session & Rows",
    [[label, ", ".join(sorted(v["captures"])), ", ".join(sorted(v["sessions"])),
      f'{v["n"]:,}'] for label, v in sorted(dates.items())])

labels = {"frozen": "Frozen IV", "surface": "Direct surface", "relaxation": "Mean reversion",
          "uncoupled": r"Noise, $\rho=0$", "coupled": "Full factor"}
audit = {(r["ticker"], r["mode"], r["check"]): r for r in read_rows("strike_audit_summary.csv")
         if r["depth"] == "401"}
rows = []
for ticker in ("GS", "LLY"):
    for mode in labels:
        a = [audit[ticker, mode, k] for k in ("bounds", "monotonicity", "vertical_spread", "convexity")]
        rows.append([ticker, labels[mode], *[f'{r["violations"]}/{r["tests"]}' for r in a],
                     f'{max(float(r["max_violation"]) for r in a):.3f}'])
write_table("strike_audit_table.tex", "llrrrrr",
            r"Ticker & Variant & Bounds & Monotonicity & Vertical & Convexity & Max (\$)", rows)

summary = read_rows("summary.csv")
rows = []
for ticker in ("GS", "LLY"):
    for kind in ("put", "call"):
        for seed in ("20260429", "20260430", "20260431"):
            r = next(x for x in summary if (x["ticker"], x["kind"], x["seed"], x["mode"], x["step"])
                     == (ticker, kind, seed, "coupled", "10"))
            rows.append([ticker + " " + kind, seed, f'{float(r["mean_abs_mark_change"]):.2f}',
                         f'{float(r["mean_pnl_change"]):+.2f}', f'{float(r["paired_se"]):.2f}'])
write_table("ablation_seeds.tex", "llrrr",
            r"Contract & Seed & Mean $|\Delta P|$ & Mean $\Delta$P\&L & Paired SE", rows)
print("Promoted ablation, strike audit, seed checks, corpus dates, and corrected scenario tables.")
