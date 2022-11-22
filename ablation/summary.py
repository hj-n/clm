
import numpy as np
import helpers as hp
import metric_run as mer
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

## argument setting

sizes = np.array([5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])
dims = dims = np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
measures_dict = {
	"ch": ["ch", "ch_range", "ch_shift", "ch_shift_range", "ch_dcal", "ch_dcal_range", "ch_dcal_shift", "ch_btw"],
	"dunn": ["dunn", "dunn_range", "dunn_shift", "dunn_shift_range", "dunn_dcal", "dunn_dcal_range", "dunn_dcal_shift", "dunn_btw"],
	"ii": ["ii", "xb", "db", "ii_range", "ii_shift", "ii_btw"],
	"sil": ["sil", "sil_btw"],
}

measures_name = {
	"ch": "Calinski-Harabasz",
	"dunn": "Dunn",
	"ii": "I-Index, Davies-Bouldin, Xie-Beni",
	"sil": "Silhouette",
}

pastel = sns.color_palette("pastel")

measures_colormap = {
	"ch": [pastel[0], pastel[1], pastel[2], pastel[3],"#1f77b4", "#f2493d", "#2ca02c", "#d62728"],
	"dunn": [pastel[0], pastel[1], pastel[2], pastel[3],"#1f77b4", "#f2493d", "#2ca02c", "#d62728"],
	"ii": ["#1f77b4", "#1f77b4", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
	"sil": ["#1f77b4", "#d62728"]
}

measures_marker = {
	"ch": ["x"] * 4 + ["o"] * 4,
	"dunn": ["x"] * 4 + ["o"] * 4,
	"ii": ["o"] * 6,
	"sil": ["o"] * 2
}

## visualize heatmaps
for measure_key in measures_dict:
	measure_list = measures_dict[measure_key]
	fig, axs = plt.subplots(2, len(measure_list), figsize=(4.5 * len(measure_list) + 9, 9))
	for i, testtype in enumerate(["shift", "card"]):
		id_array = sizes if testtype == "card" else dims
		for j, measure in enumerate(measure_list):
			scores = hp.pairwise_smape(
				f"./results_{testtype}/scores", 
				measure, id_array
			)
			print(measure, np.mean(scores))
			hp.plot_heatmap_ax(scores, id_array, axs[i, j])
	## make a common colorbar
	cb = fig.colorbar(axs[0, 0].get_children()[0], ax=axs, )
	cb.set_label("SMAPE")
	cb.outline.set_visible(False)
	# plt.tight_layout()
	plt.savefig(f"./summary_plot/{measure_key}.png", dpi=300)
	plt.savefig(f"./summary_plot/{measure_key}.pdf", dpi=300)
	plt.clf()

## visualize barplot
sns.set_style("whitegrid")
fig = plt.figure(figsize=(3.3* len(measures_dict), 6))
gs = fig.add_gridspec(2, 13)

axs = []
for i, testtype in enumerate(["shift", "card"]):
	axs_row = []
	for j, measure_key in enumerate(measures_dict):
		axs_row.append(
			fig.add_subplot(gs[i, 0:4]) if j == 0 else
			fig.add_subplot(gs[i, 4:8]) if j == 1 else
			fig.add_subplot(gs[i, 8:11]) if j == 2 else
			fig.add_subplot(gs[i, 11:13])
		)
	axs.append(axs_row)



# sns.set_style("whitegrid")
# fig, axs = plt.subplots(2, len(measures_dict), figsize=(3* len(measures_dict), 7))
for i, testtype in enumerate(["card", "shift"]):
	id_array = sizes if testtype == "card" else dims
	for j, measure_key in enumerate(measures_dict):
		hp.plot_barchart_ax(
			f"results_{testtype}/scores",
			measures_dict[measure_key],
			id_array,
			measures_name[measure_key] if i == 1 else " ",
			axs[i][j],
			measures_colormap[measure_key],
			False,
			True if j == 0 else False,
			(0, 0.28) if i == 0 else (0, 0.5),
		)
plt.subplots_adjust(hspace=0.1, wspace=0.3)

plt.savefig(f"./summary_plot/bar_2.png", dpi=300)
plt.savefig(f"./summary_plot/bar_2.pdf", dpi=300)
plt.clf()



