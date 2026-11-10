import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
out=root/"experiments/hrnet_constrained_recall_v1/comparison"
fig,axes=plt.subplots(2,2,figsize=(11,7))
for name in ["control","constrained"]:
 h=json.loads((out/name/"history.json").read_text())
 for ax,index,label in zip(axes.flat,[1,2,3,4],["Segmentation loss","Instance recall loss","Hard-background penalty","Reference retention penalty"]):
  ax.plot([r["update"] for r in h],[r["loss"][index] for r in h],label=name,alpha=.65)
  ax.set_title(label);ax.set_xlabel("Successful optimizer updates");ax.set_ylabel("Unweighted effective-batch component");ax.legend()
fig.tight_layout();fig.savefig(out/"training_components.png",dpi=150);plt.close(fig)
