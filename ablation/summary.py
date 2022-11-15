
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
	"ch": ["ch", "ch_range", "ch_shift", "ch_btw"],
	"dunn": ["dunn", "dunn_range", "dunn_shift", "dunn_btw"],
	"ii": ["ii", "ii_range", "ii_shift", "ii_btw"],
	"sil": ["sil", "sil_btw"],
	"xb": ["xb", "db", "xb_range", "xb_shift", "xb_btw"]
}
measures_name = {
	"ch": "Calinski-Harabasz",
	"dunn": "Dunn",
	"ii": "I-Index",
	"sil": "Silhouette",
	"xb": "Xie-Beni / Davies-Bouldin"
}

measures_colormap = {
	"ch": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
	"dunn": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
	"ii": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
	"sil": ["#1f77b4", "#d62728"],
	"xb": ["#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c", "#d62728"]
}

## visualize heatmaps
for measure_key in measures_dict:
	measure_list = measures_dict[measure_key]
	fig, axs = plt.subplots(2, len(measure_list), figsize=(4.5 * len(measure_list) + 4, 9))
	for i, testtype in enumerate(["shift", "card"]):
		id_array = sizes if testtype == "card" else dims
		for j, measure in enumerate(measure_list):
			scores = hp.pairwise_smape(
				f"./results_{testtype}/scores", 
				measure, id_array
			)
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
fig, axs = plt.subplots(2, len(measures_dict), figsize=(4* len(measures_dict), 8))
for i, testtype in enumerate(["shift", "card"]):
	id_array = sizes if testtype == "card" else dims
	for j, measure_key in enumerate(measures_dict):
		hp.plot_barchart_ax(
			f"results_{testtype}/scores",
			measures_dict[measure_key],
			id_array,
			measures_name[measure_key],
			axs[i, j],
			measures_colormap[measure_key],
			False
		)

plt.savefig(f"./summary_plot/bar.png", dpi=300)
plt.savefig(f"./summary_plot/bar.pdf", dpi=300)
plt.clf()



