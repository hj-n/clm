
import numpy as np
import helpers as hp
import metric_run as mer
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.lines import Line2D

## argument setting

sizes = np.array([5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])
dims = np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
testtype_arr = ["card", "shift"]

basic_ablations = ["dcal", "shift", "range", "btw"]
ablation_hier = {
	"dcal": ["dcal_shift", "dcal_range", "dcal_shift_range"],
	"shift": ["dcal_shift", "shift_range", "dcal_shift_range"],
	"range": ["dcal_range", "shift_range", "dcal_shift_range"],
	"btw": []
}
ablations = ["dcal", "shift", "range", "shift_range", "dcal_range", "dcal_shift", "dcal_shift_range"]
ablation_info = {
	"dcal": {
		False: ["ch", "ch_range", "ch_shift", "ch_shift_range", "dunn", "dunn_range", "dunn_shift", "dunn_shift_range"],
		True: ["ch_dcal", "ch_dcal_range", "ch_dcal_shift", "ch_btw", "dunn_dcal", "dunn_dcal_range", "dunn_dcal_shift", "dunn_btw"]
	},
	"range": {
		False: [
			"ch", "ch_shift", "ch_dcal", "ch_dcal_shift", 
			"dunn", "dunn_shift", "dunn_dcal", "dunn_dcal_shift",
			"ii", "xb", "db", "ii_shift"
		],
		True: [
			"ch_range", "ch_shift_range", "ch_dcal_range", "ch_btw",
			"dunn_range", "dunn_shift_range", "dunn_dcal_range", "dunn_btw",
			"ii_range", "ii_range", "ii_range", "ii_btw"
		],
	},
	"shift": {
		False: [
      "ch", "ch_range", "ch_dcal", "ch_dcal_range",
      "dunn", "dunn_range", "dunn_dcal", "dunn_dcal_range",
      "ii", "xb", "db", "ii_range",
      "sil"
		],
		True: [
      "ch_shift", "ch_shift_range", "ch_dcal_shift", "ch_btw",
      "dunn_shift", "dunn_shift_range", "dunn_dcal_shift", "dunn_btw",
      "ii_shift", "ii_shift", "ii_shift", "ii_btw",
      "sil_btw"
		],
	},
	"dcal_range": {
		False: ["ch", "ch_shift", "dunn", "dunn_shift"],
		True: ["ch_dcal_range", "ch_btw", "dunn_dcal_range", "dunn_btw"]
	},
	"dcal_shift": {
		False: ["ch", "ch_range", "dunn", "dunn_range"],
		True: ["ch_dcal_shift", "ch_btw", "dunn_dcal_shift", "dunn_btw"]
	},
	"shift_range": { 
		False: [
			"ch", "ch_dcal", "dunn", "dunn_dcal",
			"ii", "xb", "db"
		],
		True: [
			"ch_shift_range", "ch_btw", "dunn_shift_range", "dunn_btw",
			"ii_btw", "ii_btw", "ii_btw"
		]
	},
	"dcal_shift_range": {
		False: ["ch", "dunn"],
		True: ["ch_btw", "dunn_btw"]
	},
	"btw": {
		False: ["ch", "dunn", "ii", "xb", "db", "sil"],
		True: ["ch_btw", "dunn_btw", "ii_btw", "ii_btw", "ii_btw", "sil_btw"]
	}
}

ablation_info = {
	"dcal": {
		False: ["ch", "dunn"],
		True: ["ch_dcal",  "dunn_dcal" ]
	},
	"range": {
		False: [
			"ch",  
			"dunn", 
			"ii", "xb", "db", 
		],
		True: [
			"ch_range", 
			"dunn_range", 
			"ii_range", "ii_range", "ii_range",
		],
	},
	"shift": {
		False: [
      "ch", 
      "dunn", 
      "ii", "xb", "db",
      "sil"
		],
		True: [
      "ch_shift", 
      "dunn_shift", 
      "ii_shift", "ii_shift", "ii_shift", 
      "sil_btw"
		],
	},
	"dcal_range": {
		False: ["ch",  "dunn" ],
		True: ["ch_dcal_range", "dunn_dcal_range"]
	},
	"dcal_shift": {
		False: ["ch", "dunn"],
		True: ["ch_dcal_shift", "dunn_dcal_shift"]
	},
	"shift_range": { 
		False: [
			"ch",  "dunn",
			"ii", "xb", "db"
		],
		True: [
			"ch_shift_range", "dunn_shift_range", 
			"ii_btw", "ii_btw", "ii_btw"
		]
	},
	"dcal_shift_range": {
		False: ["ch", "dunn"],
		True: ["ch_btw", "dunn_btw"]
	},
	"btw": {
		False: ["ch", "dunn", "ii", "xb", "db", "sil"],
		True: ["ch_btw", "dunn_btw", "ii_btw", "ii_btw", "ii_btw", "sil_btw"]
	}
}

tab20 = sns.color_palette("tab20b").as_hex()
tab20c = sns.color_palette("tab20c").as_hex()
colormap = (
	tab20[0:2] + tab20[3:4] +
	tab20[4:6] + tab20[7:8] +
	tab20[13:14] +
	tab20c[16:17]
)

hatches = [
	None, '\\\\\\\\', '++++'  
] * 3 + [None]

colordict = {
	"dcal": colormap[0],
	"range": colormap[1],
	"shift": colormap[2],
	"dcal_range": colormap[3],
	"dcal_shift": colormap[4],
	"shift_range": colormap[5],
	"dcal_shift_range": colormap[6],
	"btw": colormap[7],
}

hatchdict = {
	"dcal": hatches[0],
	"range": hatches[1],
	"shift": hatches[2],
	"dcal_range": hatches[3],
	"dcal_shift": hatches[4],
	"shift_range": hatches[5],
	"dcal_shift_range": hatches[6],
	"btw": hatches[7],
}


markermap = {
	"ch": "o",
	"ii": "X",
	"xb": "P",
	"db": "D",
	"sil": "*",
	"dunn": "v"
}

namemap = {
	"ch": "Calinski-Harabasz",
	"ii": "I-Index",
	"xb": "Xie-Beni",
	"db": "Davies-Bouldin",
	"sil": "Silhouette",
	"dunn": "Dunn Index"
}

ablation_names = {
	"dcal": ["D-Card"] * 4,
	"range": ["Range"] * 4,
	"shift": ["Shift"] * 4,
	"dcal_range": ["D-Card & Range", "D-Card & Range", "Range & D-Card", "D-Card & Range"],
	"dcal_shift": ["D-Card & Shift", "Shift & D-Card", "D-Card & Shift", "Shift & D-Card"],
	"shift_range": ["Shift & Range", "Shift & Range", "Range & Shift", "Shift & Range"],
	"dcal_shift_range": ["D-Card & Shift & Range", "Shift & Range & D-Card", "Range & Shift & D-Card", ""],
	"btw": ["Between"] * 4
}

ablation_names_short = {
	"dcal": "D",
	"range": "R",
	"shift": "S",
	"dcal_range": "DR",
	"dcal_shift": "DS",
	"shift_range": "RS",
	"dcal_shift_range": "DRS",
	"btw": "BTW"
}

card_scores = {}
shift_scores = {}


######## DRAWING POINTPLOT BY TRICKS ########

## make subfigures
sns.set_style("whitegrid")
fig, axs = plt.subplots(
	len(testtype_arr), len(basic_ablations), 
	figsize=(len(basic_ablations)*4.2, len(testtype_arr)*4.2),
)

for i, testtype in enumerate(testtype_arr):
	id_array = sizes if testtype == "card" else dims

	for j, basic_ablation in enumerate(basic_ablations):
		ablation_list = [basic_ablation] + ablation_hier[basic_ablation]
		for k, ablation in enumerate(ablation_list):
			info = ablation_info[ablation]
			hp.plot_pointplot_ax(
				f"results_{testtype}/scores", ablation_names[basic_ablation][0],
				info[False], info[True], id_array, 
				card_scores if testtype == "card" else shift_scores,
				axs[i, j], colordict[ablation], markermap,
				True if j == 0 else False, True if i == 1 else False,
				(
					"black" if (testtype == "card" and "dcal" in ablation) or
									   (testtype == "shift" and "shift" in ablation) 
								  else None	
				)
			)
			
		## make legend
		axs[i][j].legend(
			handles=[Line2D(
				[0], [0], color=colordict[ablation_list[k]], marker='s', linestyle="None", 
				label=ablation_names[ablation_list[k]][j]
			) for k in range(len(ablation_list))],
			loc="upper left"
		)

fig.legend(
	handles=[Line2D(
		[0], [0], color="black", marker=markermap[measure], linestyle="None",
		label=namemap[measure]
	) for measure in ["ch", "ii", "xb", "db", "sil", "dunn"]],
	loc="lower center", ncol=6
)



fig.savefig(f"./summary_plot/bar_3.png", dpi=300)
fig.savefig(f"./summary_plot/bar_3.pdf", dpi=300)
plt.clf()



########### DRAWING FOR ABLATION METHOD ##############
markermap = ["o", "x", "v", "^", "s", "d", "p", "h", "8", "P"]

sns.set_style("whitegrid")
fig, axs = plt.subplots(
	1, len(testtype_arr),  
	figsize=(len(testtype_arr)*4.5, 4.5 + 1),
)

for i, testtype in enumerate(testtype_arr):
	id_array = sizes if testtype == "card" else dims
	for j, ablation in enumerate(ablation_info.keys()):
		info = ablation_info[ablation]
		before_metrics = info[False]
		after_metrics = info[True]

		hp.plot_pointplot_together_ax(
			f"results_{testtype}/scores", before_metrics, after_metrics, 
			ablation_names[ablation][0], colormap[j], markermap[j],
			card_scores if testtype == "card" else shift_scores,
			id_array, axs[i] 
		)

## place the legend under the plot
fig.legend(
	handles=[Line2D(
		[0], [0], color=colormap[j], marker=markermap[j], linestyle="None",
		label=ablation_names[ablation][0]
	) for j, ablation in enumerate(ablation_info.keys())],
	loc="lower center", ncol=3,
)


fig.savefig(f"./summary_plot/box.png", dpi=300)
fig.savefig(f"./summary_plot/box.pdf", dpi=300)

######## BOXPLOT FOR ABLATION METHOD	########

cmap = (
	tab20[0:2] + tab20[3:4] +
	tab20[4:6] + tab20[7:8] +
	tab20[13:14] +
	tab20c[16:17]
)


cmap_dict = {}
hatch_dict = {}
for i, ablation in enumerate(ablation_info.keys()):
	cmap_dict[ablation_names_short[ablation]] = cmap[i]
	hatch_dict[ablation_names_short[ablation]] = hatches[i]


sns.set_style("whitegrid")
fig, axs = plt.subplots(1, len(testtype_arr), figsize=(len(testtype_arr)*4.5, 4.5 + 1))

for i, testtype in enumerate(testtype_arr):
	id_array = sizes if testtype == "card" else dims
	hp.plot_boxplot_ax(
		f"results_{testtype}/scores", id_array, ablation_info, ablation_names_short,
		card_scores if testtype == "card" else shift_scores,
		axs[i], cmap_dict,
		True if i == 0 else False
	)


fig.savefig(f"./summary_plot/box_2.png", dpi=300)
fig.savefig(f"./summary_plot/box_2.pdf", dpi=300)

