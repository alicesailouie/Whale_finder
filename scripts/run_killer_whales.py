"""DCLDE 2027 Detection & Classification challenge: DC-1 (killer whale vs other) and
DC-2 (killer whale ecotype), evaluated on recorders never seen in training.

Usage:  python scripts/run_killer_whales.py [--epochs 12] [--folds 5]
Needs:  data/kw/features.npz and data/kw/clips.csv  (python scripts/build_kw_dataset.py)
Writes: results/killer_whales/*.csv, *.png and models/*.pt
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

from dclde import kw_data as K
from dclde import kw_model as M

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()
torch.set_num_threads(16)

OUT = ROOT / "results" / "killer_whales"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

X = np.load(K.KW_DIR / "features.npz")["X"]
clips = pd.read_csv(K.KW_DIR / "clips.csv")
groups = clips.Provider.to_numpy()
print(X.shape, clips.label.value_counts().to_dict())

ECOTYPES = ["SRKW", "TKW", "NRKW", "OKW", "SAR"]


def balanced_weights(y, n):
    c = np.bincount(y, minlength=n).astype(float)
    return np.where(c > 0, c.sum() / (n * np.maximum(c, 1)), 0.0)


def cv(task, Xs, y, g, n_out):
    """Grouped K-fold: each fold's test providers are absent from its training set."""
    proba = np.zeros((len(y), n_out))
    for k, (tr, te) in enumerate(GroupKFold(args.folds).split(Xs, y, g)):
        t = time.time()
        print(f"[{task}] fold {k + 1}: test providers {sorted(set(g[te]))}")
        model = M.train(Xs[tr], y[tr], n_out, epochs=args.epochs,
                        class_weight=balanced_weights(y[tr], n_out), seed=k, log=lambda s: None)
        proba[te] = M.predict_proba(model, Xs[te])
        print(f"   done in {time.time() - t:.0f}s")
    return proba


# ------------------------------------------------------------------ DC-1: detection
y1 = clips.is_kw.to_numpy()
p1 = cv("DC-1", X, y1, groups, 2)[:, 1]
clips["p_kw"] = p1
rows = []
for prov, g in clips.groupby("Provider"):
    r = dict(provider=prov, n=len(g), n_kw=int(g.is_kw.sum()))
    if g.is_kw.nunique() == 2:
        r.update(AUC=roc_auc_score(g.is_kw, g.p_kw), AP=average_precision_score(g.is_kw, g.p_kw))
    r["recall@0.5"] = ((g.p_kw > 0.5) & (g.is_kw == 1)).sum() / max(g.is_kw.sum(), 1)
    rows.append(r)
dc1_prov = pd.DataFrame(rows)
dc1 = dict(AUC=roc_auc_score(y1, p1), AP=average_precision_score(y1, p1),
           F1_at_0_5=f1_score(y1, p1 > 0.5), balanced_acc=balanced_accuracy_score(y1, p1 > 0.5))
# Which non-KW sounds get confused with killer whales?
neg = clips[clips.is_kw == 0]
dc1_conf = neg.groupby("label").p_kw.apply(lambda s: (s > 0.5).mean()).rename("false_alarm_rate")

# ------------------------------------------------------------------ DC-2: ecotype
kw = clips[(clips.is_kw == 1) & clips.label.isin(ECOTYPES)].reset_index()
X2 = X[kw["index"].to_numpy()]
y2 = kw.label.map({e: i for i, e in enumerate(ECOTYPES)}).to_numpy()
p2 = cv("DC-2", X2, y2, kw.Provider.to_numpy(), len(ECOTYPES))
kw["pred"] = [ECOTYPES[i] for i in p2.argmax(1)]
for i, e in enumerate(ECOTYPES):
    kw[f"p_{e}"] = p2[:, i]
st = kw[kw.label.isin(["SRKW", "TKW"])]
st_score = st.p_TKW / (st.p_TKW + st.p_SRKW)
dc2 = dict(balanced_acc_5class=balanced_accuracy_score(kw.label, kw.pred),
           SRKW_vs_TKW_AUC=roc_auc_score(st.label == "TKW", st_score),
           SRKW_vs_TKW_balanced_acc=balanced_accuracy_score(st.label == "TKW", st_score > 0.5))
cm = pd.DataFrame(confusion_matrix(kw.label, kw.pred, labels=ECOTYPES), index=ECOTYPES, columns=ECOTYPES)

# ------------------------------------------------------------------ save
clips.to_csv(OUT / "dc1_predictions.csv", index=False)
kw.drop(columns="index").to_csv(OUT / "dc2_predictions.csv", index=False)
dc1_prov.to_csv(OUT / "dc1_by_provider.csv", index=False)
cm.to_csv(OUT / "dc2_confusion.csv")
summary = pd.Series({**{f"DC1_{k}": v for k, v in dc1.items()}, **{f"DC2_{k}": v for k, v in dc2.items()},
                     **{f"DC1_false_alarm_{k}": v for k, v in dc1_conf.items()},
                     "clips": len(clips), "epochs": args.epochs, "folds": args.folds})
summary.to_csv(OUT / "summary.csv", header=["value"])
print(summary.round(3).to_string())
print(dc1_prov.round(3).to_string(index=False))
print(cm)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
dc1_prov.dropna(subset=["AUC"]).plot.barh(x="provider", y="AUC", ax=axs[0], legend=False, color="tab:blue")
axs[0].set(xlim=(0.5, 1), xlabel="ROC AUC on unseen provider", title="DC-1: killer whale detection")
cmn = cm.div(cm.sum(1).replace(0, 1), axis=0)
axs[1].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
axs[1].set(xticks=range(5), yticks=range(5), xticklabels=ECOTYPES, yticklabels=ECOTYPES,
           xlabel="predicted", ylabel="true", title="DC-2: ecotype (row-normalised)")
for i in range(5):
    for j in range(5):
        axs[1].text(j, i, f"{cmn.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn.iloc[i, j] > 0.5 else "black")
fig.tight_layout()
fig.savefig(OUT / "killer_whales.png", dpi=150)

# ------------------------------------------------------------------ final models on all data
print("training final models on all providers ...")
m1 = M.train(X, y1, 2, epochs=args.epochs, class_weight=balanced_weights(y1, 2), log=lambda s: None)
torch.save(m1.state_dict(), ROOT / "models" / "dc1_detector.pt")
m2 = M.train(X2, y2, len(ECOTYPES), epochs=args.epochs, class_weight=balanced_weights(y2, len(ECOTYPES)),
             log=lambda s: None)
torch.save(m2.state_dict(), ROOT / "models" / "dc2_ecotype.pt")
print("saved models/")
