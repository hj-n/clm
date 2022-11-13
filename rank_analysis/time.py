import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl

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
	"CH_btwn", "Dunn_btwn", "II_btwn", "Sil_btwn", "XB_btwn",
	"SVM", "KNN", "MLP", "NB", "RF", "LR", "LDA",
]



clusterings = [
	"birch",
	"hdbscan",
	"kmeans",
	"xmeans",
	"dbscan",
	# "kmedoid",
	"agglo_average",
	"agglo_complete",
	"agglo_single",
]

## make a colormap that shares same rgb with differnet opacity for each line
colors = (
 [mpl.colors.rgb2hex((0.12, 0.46, 0.70, alpha)) for alpha in np.linspace(0.2, 1, 6)] +
 [mpl.colors.rgb2hex((0.85, 0.37, 0.01, alpha), keep_alpha=True) for alpha in np.linspace(0.2, 1, 5)] +
 [mpl.colors.rgb2hex((0.49, 0.18, 0.56, alpha), keep_alpha=True) for alpha in np.linspace(0.2, 1, 7)] +
 [mpl.colors.rgb2hex((0.47, 0.67, 0.19, alpha), keep_alpha=True) for alpha in np.linspace(0.2, 1, 8)]
)



clusterings_name = [
	"BIRCH", 
	"HDBSCAN", 
	"K-Means",
	"X-Means",
	"DBSCAN",
	# "K-Medoid",
	"Agglo (Average)",
	"Agglo (Complete)",
	"Agglo (Single)",
]

time_df = pd.DataFrame({
	"measure": [],
	"score": []
})

for i, measure in enumerate(measures):
	with open(f"results/measures/{measure}_time.json") as file:
		times = json.load(file)
		time_df = time_df.append(pd.DataFrame({
			"measure": [measures_name[i]] * len(times),
			"score": times
		}))

		print(f"{measure}: {np.sum(np.array(times))}")

for i, clustering in enumerate(clusterings):
	with open(f"results/clusterings/{clustering}_ami_time.json") as file:
		times = json.load(file)
		time_df = time_df.append(pd.DataFrame({
			"measure": [clusterings_name[i]] * len(times),
			"score": times
		}))
		print(f"{clustering}: {np.sum(np.array(times))}")


plt.figure(figsize=(7, 8))
sns.set(style="whitegrid")


ax = sns.barplot(x="score", y="measure", data=time_df, palette=colors)

## set y to be log
ax.set_xscale("log")


plt.tight_layout()

plt.savefig("results/summary/time.png", dpi=300)
plt.savefig("results/summary/time.pdf", dpi=300)
