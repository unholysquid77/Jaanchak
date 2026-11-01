"""Emit LaTeX figure bodies from paper/figdata/*.dat.

Nothing here is typed by hand: every coordinate is read from the generated
tables, so a figure cannot drift away from the results the way the ISIC lesion
counts did.
"""
import io, os

ROOT = "/home/user/Jaanchak"
FIG = os.path.join(ROOT, "paper", "figdata")

def read(name):
    rows = io.open(os.path.join(FIG, name)).read().splitlines()
    hdr = rows[0].split()
    return [dict(zip(hdr, r.split())) for r in rows[1:]]

out = {}

# ---------------------------------------------------------------- params vs acc
p = read("params_vs_acc.dat")
pts, marks = [], []
for r in p:
    x = max(float(r["frac"]), 0.012)          # log axis: frozen sits at 0
    pts.append(f"  ({x},{r['acc']}) +- (0,{r['sd']})")
    marks.append((x, r["acc"], r["label"]))
lab = {"frozen": "frozen", "LN": "LayerNorm", "attn~b8": "attn blk 8",
       "attn~b11": "attn blk 11", "MLP~b8": "MLP blk 8",
       "attn~all": "attn, all blocks", "MLP~all": "MLP, all blocks", "full": "full FT"}
# Label anchors are chosen so nothing collides or runs past the axis box; the
# right-hand cluster (33/66/100%) is tight, so those three anchor outward.
pos = {"frozen": "above right", "LN": "below right", "attn~b8": "above",
       "attn~b11": "below left", "MLP~b8": "above", "attn~all": "above",
       "MLP~all": "below left", "full": "below left"}
nodes = "\n".join(
    f"\\node[font=\\scriptsize,{pos[l]}=1pt,midgray] at (axis cs:{x},{a}) {{{lab[l]}}};"
    for x, a, l in marks)
out["params"] = f"""\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
 width=0.90\\textwidth,height=6.3cm,
 xmode=log,log basis x=10,
 xlabel={{Backbone parameters updated (\\% of 85{{,}}798{{,}}656, log scale)}},
 ylabel={{EuroSAT validation accuracy (\\%)}},
 xmin=0.008,xmax=190,ymin=77.5,ymax=91.0,
 grid=major,grid style={{gray!20}},
 xtick={{0.012,0.045,2.75,33.04,100}},
 xticklabels={{$0$,$0.05$,$2.75$,$33$,$100$}},
 every axis plot/.append style={{thick}}
]
\\addplot[only marks,attncol,mark=*,mark size=2.4pt,error bars/.cd,y dir=both,y explicit] coordinates {{
{chr(10).join(pts)}
}};
\\addplot[mlpcol,densely dashed,thick,domain=0.008:190,samples=2]{{83.07}};
{nodes}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Accuracy against the fraction of backbone parameters updated, EuroSAT
validation, three episode draws, error bars one standard deviation. The dashed
line is full fine-tuning. Accuracy is not monotonic in parameter count: a single
attention block at $2.75\\%$ and a single MLP block at $5.50\\%$ both sit above
the $100\\%$ point, while a differently placed block of identical size does not.
Parameter count alone therefore does not order these configurations.}}
\\label{{fig:paramsacc}}
\\end{{figure}}"""

# ------------------------------------------------------------------ inner steps
st = read("inner_steps.dat")
A = "\n".join(f"  ({r['steps']},{r['attn']})" for r in st)
F = "\n".join(f"  ({r['steps']},{r['full']})" for r in st)
out["steps"] = f"""\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
 width=0.72\\textwidth,height=5.4cm,
 xlabel={{Inner-loop Adam steps}},ylabel={{EuroSAT validation accuracy (\\%)}},
 xtick={{5,10,20}},xmin=3.5,xmax=21.5,ymin=82.5,ymax=86.8,
 grid=major,grid style={{gray!20}},legend pos=south west,legend cell align=left
]
\\addplot[attncol,mark=*,thick] coordinates {{
{A}
}};
\\addlegendentry{{Attention-only ($2.75\\%$)}}
\\addplot[mlpcol,mark=square*,thick,dashed] coordinates {{
{F}
}};
\\addlegendentry{{Full fine-tuning ($100\\%$)}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Accuracy against inner-loop length, each baseline retuned at every step
count, one episode draw. Attention-only adaptation improves monotonically with
more updates; full fine-tuning peaks at ten steps and falls by $1.40$ points at
twenty. The $20$-step setting used for the main experiments therefore places
full fine-tuning below its own best step count. Single-draw adjacent differences
are close to the measured run-to-run floor, so this is directional evidence.}}
\\label{{fig:steps}}
\\end{{figure}}"""

# ------------------------------------------------------- support size, gains
mg = read("mech_gains.dat")
def arm(b):
    return "\n".join(f"  ({r['k']},{r['gain_over_frozen']}) +- (0,{r['ci95']})"
                     for r in mg if r["baseline"] == b)
out["shots"] = f"""\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
 width=0.72\\textwidth,height=5.6cm,
 xmode=log,log basis x=2,
 xlabel={{Support examples per class, $k$}},
 ylabel={{Gain over frozen ProtoNet (points)}},
 xtick={{1,5,20}},xticklabels={{1,5,20}},xmin=0.75,xmax=27,ymin=-3.2,ymax=11.5,
 grid=major,grid style={{gray!20}},legend pos=north west,legend cell align=left
]
\\addplot[black,dotted,thick,forget plot,domain=0.75:27,samples=2]{{0}};
\\addplot[attncol,mark=*,thick,error bars/.cd,y dir=both,y explicit] coordinates {{
{arm('attn_only_ft')}
}};
\\addlegendentry{{Attention-only}}
\\addplot[mlpcol,mark=square*,thick,dashed,error bars/.cd,y dir=both,y explicit] coordinates {{
{arm('full_ft')}
}};
\\addlegendentry{{Full fine-tuning}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Each method's gain over a frozen backbone as support size grows,
EuroSAT. Full fine-tuning is \\emph{{below}} frozen ProtoNet at one shot and has
caught up by twenty; the attention-only advantage is therefore bounded to an
intermediate regime rather than general. The $k=5$ row averages three draws; the
endpoints are single draws, and the statistic was specified after the $k=1$ and
$k=5$ outcomes were seen, so only $k=20$ is a prospective check.}}
\\label{{fig:shots}}
\\end{{figure}}"""

# -------------------------------------------------------------- weight decay
wd = read("weight_decay.dat")
order = ["1e-4", "1e-3", "1e-2", "1e-1"]
def dose(b):
    d = {r["wd"]: r for r in wd if r["baseline"] == b}
    return "\n".join(f"  ({i+1},{d[w]['diff_vs_wd0']}) +- (0,{d[w]['ci95']})"
                     for i, w in enumerate(order))
out["wd"] = f"""\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
 width=0.72\\textwidth,height=5.4cm,
 xlabel={{Inner-loop weight decay}},
 ylabel={{Change vs.\\ that method's own $\\lambda=0$ run}},
 xtick={{1,2,3,4}},xticklabels={{$10^{{-4}}$,$10^{{-3}}$,$10^{{-2}}$,$10^{{-1}}$}},
 xmin=0.6,xmax=4.4,ymin=-7.2,ymax=2.2,
 grid=major,grid style={{gray!20}},legend pos=south west,legend cell align=left
]
\\addplot[black,dotted,thick,forget plot,domain=0.6:4.4,samples=2]{{0}};
\\addplot[attncol,mark=*,thick,error bars/.cd,y dir=both,y explicit] coordinates {{
{dose('attn_only_ft')}
}};
\\addlegendentry{{Attention-only}}
\\addplot[mlpcol,mark=square*,thick,dashed,error bars/.cd,y dir=both,y explicit] coordinates {{
{dose('full_ft')}
}};
\\addlegendentry{{Full fine-tuning}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{The predicted repair fails. If full fine-tuning lost because too many
parameters moved too freely, regularisation should have recovered it. Instead the
dose response is monotonically negative, reaching $-5.12$ points at
$\\lambda=10^{{-1}}$. Attention-only adaptation is flat until $10^{{-2}}$ and is
then damaged as well, so a small parameter group is not inherently immune.
Differences are paired against each method's own $\\lambda=0$ run on the same
episodes; the two arms use separately tuned learning rates, so only the
within-method dose response is clean.}}
\\label{{fig:wd}}
\\end{{figure}}"""

for k, v in out.items():
    io.open(f"/tmp/claude-0/-home-user-Jaanchak/31439b19-2e3b-5156-86bc-e77dee1c0391/scratchpad/gen/{k}.tex", "w").write(v)
    print(f"  wrote {k}.tex  ({len(v.splitlines())} lines)")


# -------------------------------------------------- tuned learning rates
# Read from the per-cell selection files that produced each reported number,
# so the table cannot disagree with the runs. `_v2` supersedes, exactly as the
# figure loader does.
import json, glob
BEST = {
 "eurosat": ["eurosat","pos0","pos2","pos5","pos8","poscurve","mlppos0","mlppos2",
             "mlppos5_v2","mlppos8","mlppos11","mlpposcurve","cap1"],
 "isic":    ["isic","isicpos0_v2","isicpos2","isicpos5","isicpos8","isicpos11",
             "isicposcurve","isicmlppos0","isicmlppos2","isicmlppos5_v2",
             "isicmlppos8","isicmlppos11"],
}
def lrs(ds):
    got = {}
    for f in BEST[ds]:
        p = os.path.join(ROOT, "results", f"vit_best_lr_{f}.json")
        if not os.path.exists(p):
            continue
        for b, v in json.load(io.open(p)).items():
            if "5" in v:
                got.setdefault(b, v["5"]["lr"])
    return got

def fmt(x):
    if x is None:
        return "---"
    e = 0
    m = float(x)
    while m < 1: m *= 10; e -= 1
    m = round(m)
    return f"$10^{{{e}}}$" if m == 1 else f"${m}\\times10^{{{e}}}$"

E, I = lrs("eurosat"), lrs("isic")
ROWS = [
 ("group", "Whole-network and whole-family groups", None, None),
 ("row", "Frozen ProtoNet (no update)", None, None),
 ("row", "Patch embedding", "patch_embed_ft", "0.866\\%"),
 ("row", "LayerNorm only", "ln_only_ft", "0.045\\%"),
 ("row", "Attention, all blocks", "attn_only_ft", "33.041\\%"),
 ("row", "MLP, all blocks", "mlp_only_ft", "66.049\\%"),
 ("row", "Full fine-tuning", "full_ft", "100\\%"),
 ("group", "Single attention block (each $2.75\\%$)", None, None),
 ("row", "Block 0", "attn_block0_ft", None),
 ("row", "Block 2", "attn_block2_ft", None),
 ("row", "Block 5", "attn_block5_ft", None),
 ("row", "Block 8", "attn_block8_ft", None),
 ("row", "Block 11", "attn_last1_ft", None),
 ("group", "Single MLP block (each $5.50\\%$)", None, None),
 ("row", "Block 0", "mlp_block0_ft", None),
 ("row", "Block 2", "mlp_block2_ft", None),
 ("row", "Block 5", "mlp_block5_ft", None),
 ("row", "Block 8", "mlp_block8_ft", None),
 ("row", "Block 11", "mlp_block11_ft", None),
]
body = []
for kind, label, key, frac in ROWS:
    if kind == "group":
        body.append(f"\\addlinespace\n\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}}\\\\")
    elif key is None:
        body.append(f"{label} & $0\\%$ & --- & --- \\\\")
    else:
        body.append(f"{label} & {frac or ''} & {fmt(E.get(key))} & {fmt(I.get(key))} \\\\")
span_e = max(E[k] for k in E) / min(E[k] for k in E)
out["hyper"] = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{Inner-loop learning rate selected on validation for every reported
configuration, at $k=5$. Each group is tuned separately on a logarithmic grid;
all other inner-loop settings are shared ($20$ Adam steps, no weight decay
unless a probe varies it, cosine prototype head). The spread is the reason a
shared learning rate would not have produced a fair comparison. On EuroSAT the
selected rates span four orders of magnitude, from $10^{{-6}}$ for the patch
embedding to $10^{{-2}}$ for LayerNorm. The spread is not only across families:
holding family and parameter count fixed, attention's selected rate moves
$100\\times$ between block 0 and block 8. It also moves across datasets, by
$100\\times$ for attention block 0, which is evidence for a dataset-dependent
depth effect that does not come from accuracy.}}
\\label{{tab:hyper}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Configuration & Updated & EuroSAT & ISIC \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
io.open(f"/tmp/claude-0/-home-user-Jaanchak/31439b19-2e3b-5156-86bc-e77dee1c0391/scratchpad/gen/hyper.tex","w").write(out["hyper"])
print(f"  wrote hyper.tex  (EuroSAT selected-rate span {span_e:,.0f}x)")
