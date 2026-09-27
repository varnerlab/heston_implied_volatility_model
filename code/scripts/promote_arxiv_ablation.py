"""Copy verified experiment outputs and generate supporting arXiv tables.

Pass --tables-only to regenerate the tables without recopying the scenario
tables, the ablation table, or the ablation figure (the manuscript figure is
re-rendered separately and must not be overwritten).
"""
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "code/results/dynamic_ablation"
SECTIONS = ROOT / "paper-arxiv/sections"
GENERATED = SECTIONS / "generated"
GENERATED.mkdir(exist_ok=True)
if "--tables-only" not in sys.argv:
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

# Violation counts and largest deviations for each check at both tree depths.
checks = {"bounds": "Bounds", "monotonicity": "Monotonicity",
          "vertical_spread": "Vertical", "convexity": "Convexity"}
by_depth = {(r["ticker"], r["mode"], r["check"], r["depth"]): r
            for r in read_rows("strike_audit_summary.csv")}
rows = []
for ticker in ("GS", "LLY"):
    for mode in labels:
        for check, name in checks.items():
            a, b = by_depth[ticker, mode, check, "201"], by_depth[ticker, mode, check, "401"]
            rows.append([ticker, labels[mode], name, a["violations"], b["violations"], b["tests"],
                         f'{float(a["max_violation"]):.3f}', f'{float(b["max_violation"]):.3f}'])
write_table("strike_audit_depth_table.tex", "lllrrrrr",
            r"Ticker & Variant & Check & Viol.\ 201 & Viol.\ 401 & Tests & Max 201 (\$) & Max 401 (\$)",
            rows)

summary = read_rows("summary.csv")
# Path-date diagnostics depend on the stock paths, not the IV variant; the floor
# fraction is taken as the maximum across variants.
diagnostics = defaultdict(lambda: {"outside": set(), "floor": 0.0})
for r in read_rows("factor_diagnostics.csv"):
    d = diagnostics[r["ticker"], r["kind"], r["seed"]]
    d["outside"].add(float(r["outside_moneyness_fraction"]))
    d["floor"] = max(d["floor"], float(r["floor_fraction"]))
# Largest 201- versus 401-step price change across variants and checked dates.
tree_change = defaultdict(float)
for r in read_rows("numerical_check.csv"):
    key = (r["ticker"], r["kind"], r["seed"])
    tree_change[key] = max(tree_change[key], float(r["absolute_change"]))
rows = []
for ticker in ("GS", "LLY"):
    for kind in ("put", "call"):
        for seed in ("20260429", "20260430", "20260431"):
            r = next(x for x in summary if (x["ticker"], x["kind"], x["seed"], x["mode"], x["step"])
                     == (ticker, kind, seed, "coupled", "10"))
            d = diagnostics[ticker, kind, seed]
            assert len(d["outside"]) == 1, "outside-interval share should not depend on the variant"
            rows.append([ticker + " " + kind, seed, f'{float(r["mean_abs_mark_change"]):.2f}',
                         f'{float(r["mean_pnl_change"]):+.2f}', f'{float(r["paired_se"]):.2f}',
                         f'{100 * d["outside"].pop():.1f}', f'{100 * d["floor"]:.1f}',
                         f'{tree_change[ticker, kind, seed]:.4f}'])
write_table("ablation_seeds.tex", "llrrrrrr",
            r"Contract & Seed & Mean $|\Delta P|$ & Mean $\Delta$P\&L & Paired SE"
            r" & Outside (\%) & Floor (\%) & Tree change (\$)", rows)
print("Promoted ablation, strike audit, seed checks, corpus dates, and corrected scenario tables.")
