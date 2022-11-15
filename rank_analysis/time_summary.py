import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multitest import multipletests

## disable warning
import warnings
warnings.filterwarnings("ignore")


measures = [
	"ch","dunn", "db", "ii",  "sil","xb", 
	"ch_btw", "dunn_btw", "ii_btw", "sil_btw", "xb_btw",
	"svm", "knn", "mlp", "nb", "rf", "lr", "lda", 
]

measures_name = [
	"CH", "Dunn", "DB", "II",  "Sil", "XB", 
	"CH_btwn", "Dunn_btwn", "II_btwn", "Sil_btwn", "XB_btwn (=DB_btwn)",
	"SVM", "KNN", "MLP", "NB", "RF", "LR", "LDA",
]



clusterings = [
	"birch",
	"hdbscan",
	"kmeans",
	"xmeans",
	"dbscan",
	"kmedoid",
	"agglo_average",
	"agglo_complete",
	"agglo_single",
]

## make a colormap that shares same rgb with differnet opacity for each line
colors = (
 [(0.12 + alpha, 0.46 + alpha, min(0.70 + alpha,1)) for alpha in np.linspace(0, 0.4, 6)] +
 [(min(0.85 + alpha, 1), 0.37 + alpha, 0.01 + alpha) for alpha in np.linspace(0, 0.4, 5)] +
 [(0.49 + alpha, 0.18 + alpha, min(0.56 + alpha,1)) for alpha in np.linspace(0, 0.4, 7)] +
 [(0.47 + alpha, min(0.67 + alpha, 1), 0.19 + alpha) for alpha in np.linspace(0, 0.4, 9)]
)



clusterings_name = [
	"BIRCH", 
	"HDBSCAN", 
	"K-Means",
	"X-Means",
	"DBSCAN",
	"K-Medoid",
	"Agglo (Average)",
	"Agglo (Complete)",
	"Agglo (Single)",
]

time_df = pd.DataFrame({
	"measurement": [],
	"time": []
})

for i, measure in enumerate(measures):
	with open(f"results/measures/{measure}_time.json") as file:
		times = json.load(file)
		time_df = time_df.append(pd.DataFrame({
			"measurement": [measures_name[i]] * len(times),
			"time": times
		}))

		print(f"{measure}: {np.sum(np.array(times))}")

for i, clustering in enumerate(clusterings):
	with open(f"results/clusterings/{clustering}_ami_time.json") as file:
		times = json.load(file)
		time_df = time_df.append(pd.DataFrame({
			"measurement": [clusterings_name[i]] * len(times),
			"time": times
		}))
		print(f"{clustering}: {np.sum(np.array(times))}")


plt.figure(figsize=(6.5, 8))
sns.set(style="whitegrid")


ax = sns.boxplot(
	x="time", y="measurement", data=time_df, palette=colors,
	flierprops={"marker": "x"}	
)

## set y to be log
ax.set_xscale("log")
plt.tight_layout()
plt.savefig("results/summary/time.png", dpi=300)
plt.savefig("results/summary/time.pdf", dpi=300)
