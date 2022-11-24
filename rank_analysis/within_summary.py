
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns


pairs = [
  ("ch", "ch_btw"),
  ("dunn", "dunn_btw"),
  ("ii", "ii_btw"),
  ("xb", "ii_btw"),
  ("db", "ii_btw"),
  # ("sil", "sil_btw"),
]

pname = [
	("CH", "CH_btw"),
	("Dunn", "Dunn_btw"),
	("II", "II_btw"),
	("XB", "XB_btw"),
	("DB", "DB_btw"),
	# ("Sil", "Sil_btw"),
]



DATASET_LIST = np.load("./results/dataset_list.npy")


x_criteria = "ch_btw"
x_criteria_name = "CH_btw score"
with open(f"./results/measures/{x_criteria}_score.json", "r") as f:
	x_data = json.load(f)



rho_arr  = []
pair_arr = []
x_arr = []
type_arr = []
for pidx, pair in enumerate(pairs):
	rho_12_temp = []
	rho_gt2_temp = []
	pair_12_temp = []
	pair_gt2_temp = []
	for dataset in DATASET_LIST:
		score_1 = np.load("./within_results/scores/{}/{}.npy".format(pair[0], dataset))
		if pair[0] == "xb" or pair[0] == "db":
			score_1 = -score_1
		score_2 = np.load("./within_results/scores/{}/{}.npy".format(pair[1], dataset))
		score_gt = np.array([10, 9, 8, 7, 6 ,5, 4, 3, 2, 1, 0])
		
		rho_12, _ = spearmanr(score_1, score_2)
		rho_gt2, _ = spearmanr(score_gt, score_2)
		rho_12_temp.append(rho_12)
		rho_gt2_temp.append(rho_gt2)
		pair_12_temp.append(f"{pname[pidx][1]}")
		pair_gt2_temp.append(f"{pname[pidx][1]}")

	## filter NaN for rho_12_temp
	filtering = np.logical_not(np.isnan(rho_12_temp))
	rho_12_temp = np.array(rho_12_temp)[filtering].tolist()
	x_12_temp = np.array(x_data)[filtering].tolist()
	pair_12_temp = np.array(pair_12_temp)[filtering].tolist()

	## filter NaN for rho_gt2_temp
	filtering = np.logical_not(np.isnan(rho_gt2_temp))
	rho_gt2_temp = np.array(rho_gt2_temp)[filtering].tolist()
	x_gt2_temp = np.array(x_data)[filtering].tolist()
	pair_gt2_temp = np.array(pair_gt2_temp)[filtering].tolist()


	print(f"{pname[pidx][0]}-{pname[pidx][1]}: {np.mean(rho_12_temp)}")
	print(f"GT-{pname[pidx][1]}: {np.mean(rho_gt2_temp)}")


	x_arr += x_12_temp + x_gt2_temp
	rho_arr += rho_12_temp + rho_gt2_temp
	pair_arr += pair_12_temp + pair_gt2_temp
	type_arr += ["IVM_btwn vs IVM"] * len(rho_12_temp) + ["IVM_btwn vs GT"] * len(rho_gt2_temp)
	

df = pd.DataFrame({"pair": pair_arr, "rho": rho_arr, x_criteria_name: x_arr, "type": type_arr})


## draw boxplot
sns.set_theme(style="whitegrid")
# ax = sns.boxplot(x=x_criteria, y="rho", hue="pair", data=df, palette="Set3", order=binning_ranges[:-1])

fg = sns.FacetGrid(
	df, row="type",col="pair", hue="pair", height=2, aspect=1.2,	palette="tab10"
)
fg.map_dataframe(
	sns.regplot, x=x_criteria_name, y="rho", lowess=True, ci=95, scatter_kws={"s": 20, "alpha": 0.5}, marker="+",
	line_kws={"color": "black", "lw": 1, "alpha": 0.8, "linestyle": "--"}
)

axes = fg.axes.flatten()
print(axes)
for i, ax in enumerate(axes):
	if i == 0:
		ax.set_ylabel("Correlation with IVM")
	elif i == len(pname):
		ax.set_ylabel("Correlation with GT")
	
	if i < len(pname):
		ax.set_title(pname[i][1])
	else:
		ax.set_title("")
# for i, pn in enumerate(pname):
# 	axes[i].set_title(pn[0] + " / " + pn[1])

# .map(
# 	sns.regplot, x=x_criteria, y="rho", lowess=True, scatter_kws={"s": 5}
# )


## reverse x axis
plt.gca().invert_xaxis()

## make x axis tick to show only two 

## make y axis log


plt.savefig("./within_results/plots/result.png", dpi=300)
plt.savefig("./within_results/plots/result.pdf", dpi=300)

	