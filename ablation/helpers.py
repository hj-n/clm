import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import explained_variance_score

def check_and_make(path):
	if not os.path.exists(path):
		os.makedirs(path)
	
def read_json(path):
	with open(path, 'r') as f:
		data = json.load(f)
	return np.array(data)

def save_json(data, file_name):
	"""
	save the testing result to a json file
	"""
	with open(file_name, "w") as f:
		json.dump(data, f)


def weighted_smape(path, metric, weight, id1, id2):
	path_1 = f"{path}/{metric}/{id1}.json"
	path_2 = f"{path}/{metric}/{id2}.json"

	data_1 = np.array(read_json(path_1)[:])
	data_2 = np.array(read_json(path_2)[:])

	# min_12 = max(min(np.min(data_1), np.min(data_2)), 0)
	min_12 = min(np.min(data_1), np.min(data_2))


	data_1 = (data_1 - min_12) 
	data_2 = (data_2 - min_12) 

	# error_arr = []
	# for i in range(data_1.shape[0]):
	# 	if data_1[i] == 0 and data_2[i] == 0:
	# 		error_arr.append(0)
	# 	else:
	# 		error_arr.append((weight[i] *np.abs(data_1[i] - data_2[i])) / (np.abs(data_1[i]) + np.abs(data_2[i])))

	# return np.mean(error_arr)

	smape_nominator = 0
	smape_denominator = 0
	for i in range(data_1.shape[0]):
		smape_nominator += np.abs(data_1[i] - data_2[i]) * weight[i]
		smape_denominator += np.abs(data_1[i]) + np.abs(data_2[i])
	
	if (smape_denominator == 0):
		return 0

	return smape_nominator / smape_denominator



def pairwise_weighted_smape(path, metric, id_array):
	## compute weight
	binning = 10
	binspace = np.linspace(0, 1, binning+1)[:binning]
	binweights = np.zeros(binspace.shape, dtype=np.int32)
	bininterval = 1.0 / binning

	metadata = pd.read_csv("./data_metadata/1000_2gaussians_proba_judgment_ClustMe_EXP1.csv")
	file_keys = metadata["XYposCSVfilename"]
	human_judgments = metadata["probMore"]
	for i in range(len(human_judgments)):
		idx = int(human_judgments[i] // bininterval)
		if idx == binning:
			idx -= 1
		binweights[idx] += 1
	## normalize
	binweights = 1 / binweights
	binweights = binweights / np.sum(binweights)

	key_to_weights = {}
	for i in range(len(file_keys)):
		idx = int(human_judgments[i] // bininterval)
		if idx == binning:
			idx -= 1
		key_to_weights[file_keys[i]] = binweights[idx]
	sorted_file_keys = sorted(file_keys)
	## compute weight list
	weight_list = []
	for i in range(len(sorted_file_keys)):
		weight_list.append(key_to_weights[sorted_file_keys[i]])

		
	scores = np.zeros((id_array.shape[0], id_array.shape[0]))
	for i, id1 in enumerate(id_array):
		for j, id2 in enumerate(id_array):
			scores[i, j] = weighted_smape(path, metric, weight_list, id1, id2)

	return scores

def smape_data(data_1, data_2):
	"""
	compute the symmetric mean absolute percentage error
	"""
	# min_12 = max(min(np.min(data_1), np.min(data_2)), 0)
	min_12 = min(np.min(data_1), np.min(data_2))

	data_1 = (data_1 - min_12) 
	data_2 = (data_2 - min_12) 

	smape_nominator = 0
	smape_denominator = 0
	for i in range(data_1.shape[0]):
		smape_nominator += np.abs(data_1[i] - data_2[i])
		smape_denominator += np.abs(data_1[i]) + np.abs(data_2[i])

	if (smape_denominator == 0):
		return 0
	result = smape_nominator / smape_denominator

def smape(path, metric, id1, id2):
	path_1 = f"./{path}/{metric}/{id1}.json"
	path_2 = f"{path}/{metric}/{id2}.json"

	data_1 = np.array(read_json(path_1)[:])
	data_2 = np.array(read_json(path_2)[:])

	max_12 = max(np.max(data_1), np.max(data_2))
	min_12 = min(np.min(data_1), np.min(data_2))

	data_1 = (data_1 - min_12) 
	data_2 = (data_2 - min_12) 
	data_reverse_1 = max_12 - data_1
	data_reverse_2 = max_12 - data_2

	## method 1
	
	# error_arr = []
	# for i in range(data_1.shape[0]):
	# 	if data_1[i] == 0 and data_2[i] == 0:
	# 		error_arr.append(0)
	# 	else:
	# 		error_arr.append(np.abs(data_1[i] - data_2[i]) / (np.abs(data_1[i]) + np.abs(data_2[i])))
	# return (np.mean(error_arr))
	# reverse_error_arr = []
	# for i in range(data_1.shape[0]):
	# 	if data_reverse_1[i] == 0 and data_reverse_2[i] == 0:
	# 		reverse_error_arr.append(0)
	# 	else:
	# 		reverse_error_arr.append(np.abs(data_reverse_1[i] - data_reverse_2[i]) / (np.abs(data_reverse_1[i]) + np.abs(data_reverse_2[i])))


	## method 2

	smape_nominator = 0
	smape_denominator = 0
	for i in range(data_1.shape[0]):
		smape_nominator += np.abs(data_1[i] - data_2[i])
		smape_denominator += np.abs(data_1[i]) + np.abs(data_2[i])

	if (smape_denominator == 0):
		return 0
	result = smape_nominator / smape_denominator

	# smape_nominator = 0
	# smape_denominator = 0
	# for i in range(data_1.shape[0]):
	# 	smape_nominator += np.abs(data_reverse_1[i] - data_reverse_2[i])
	# 	smape_denominator += np.abs(data_reverse_1[i]) + np.abs(data_reverse_2[i])
	# reverse_result = smape_nominator / smape_denominator


	return result


def pairwise_smape(path, metric, id_array):
	scores = np.zeros((id_array.shape[0], id_array.shape[0]))
	for i, id1 in enumerate(id_array):
		for j, id2 in enumerate(id_array):
			scores[i, j] = smape(path, metric, id1, id2)

	return scores

def rsq(path, metric, id1, id2):
	path_1 = f"{path}/{metric}/{id1}.json"
	path_2 = f"{path}/{metric}/{id2}.json"

	data_1 = read_json(path_1)[:]
	data_2 = read_json(path_2)[:]
	r2_1 = explained_variance_score(data_1, data_2)
	r2_2 = explained_variance_score(data_2, data_1)
	return np.mean([r2_1, r2_2])

def pairwise_rsq(path, metric, id_array):
	scores = np.zeros((id_array.shape[0], id_array.shape[0]))
	for i, id1 in enumerate(id_array):
		for j, id2 in enumerate(id_array):
			scores[i, j] = rsq(path, metric, id1, id2)

	return scores

### random_array_generator (float)
def random_array(size, min_val, max_val):
	return np.random.uniform(min_val, max_val, size)

def random_array_int(size, min_val, max_val):
	return np.random.randint(min_val, max_val + 1, size)


def plot_barchart(path, metrics, id_array, type):
	score_arry = []
	for metric in metrics:
		scores = pairwise_weighted_smape(path, metric, id_array)
		mean_scores = np.sum(scores) / ((scores.shape[0] - 1) * scores.shape[0])
		score_arry.append(mean_scores)

	df = pd.DataFrame({'measure': metrics, 'SMAPE': score_arry})

	plt.figure(figsize=(6, 6))
	sns.set(style="whitegrid")
	ax = sns.barplot(x="measure", y="SMAPE", data=df)

	plt.tight_layout()
	check_and_make(f"./results_{type}/plots")
	plt.savefig(f"./results_{type}/plots/summary.png")

	plt.clf()

	
	

def plot_pointplot_ax(
	# path, ablation_arr, info_arr, id_array, scores, ax
	path, ablation, before_metrics, after_metrics, id_array, scores, ax, color, markermap,
	show_yLabel, show_xLabel, border, lim
):

	before_scores_array = []
	after_scores_array = []
	metrics_array = []
	for i in range(len(before_metrics)):
		if before_metrics[i] in scores:
			before_scores = scores[before_metrics[i]]
		else:
			before_scores = pairwise_weighted_smape(path, before_metrics[i], id_array).flatten()
			scores[before_metrics[i]] = before_scores
		if after_metrics[i] in scores:
			after_scores = scores[after_metrics[i]]
		else:
			after_scores = pairwise_weighted_smape(path, after_metrics[i], id_array).flatten()
			scores[after_metrics[i]] = after_scores

		before_scores_array += before_scores.tolist()
		after_scores_array += after_scores.tolist()
		metrics_array += [before_metrics[i]] * len(before_scores)

	df = pd.DataFrame({
		'measure': metrics_array,
		'before': before_scores_array,
		'after': after_scores_array,
	})

	for measure in before_metrics:
		measure_df = df[df['measure'] == measure]
		before_mean = np.mean(measure_df['before'])
		after_mean = np.mean(measure_df['after'])
		before_std = np.std(measure_df['before'])
		after_std = np.std(measure_df['after'])
		before_ci = 1.96 * before_std / np.sqrt(len(measure_df['before']))
		after_ci = 1.96 * after_std / np.sqrt(len(measure_df['after']))
		ax.errorbar(before_mean, after_mean, xerr=before_ci, yerr=after_ci, c=color, zorder=1)
		ax.scatter(before_mean, after_mean, 
			s=250 if markermap[measure] == '+' or markermap[measure] == "x" or markermap[measure] == "*" else 100,
			c=color, marker=markermap[measure], edgecolor=border, 
			linewidth=1.3, zorder=10
		)

	ax.axline((0, 0), slope=1, c='red', linestyle='--')
	ax.axis('square')

	ax.set_xlim(lim)
	ax.set_ylim(lim)


	## scientific notation
	ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

	if show_yLabel:
		ax.set_ylabel('After Applying Tricks')
	if show_xLabel:
		ax.set_xlabel('Before Applying Tricks')


def plot_pointplot_together_ax(path, before_metrics, after_metrics, ablation_name, color, marker, scores, id_array, ax):
	before_scores_arr = []
	after_scores_arr  = []
	for i in range(len(before_metrics)):
		if before_metrics[i] in scores:
			before_scores = scores[before_metrics[i]]
		else:
			before_scores = pairwise_weighted_smape(path, before_metrics[i], id_array).flatten()
			scores[before_metrics[i]] = before_scores
		if after_metrics[i] in scores:
			after_scores = scores[after_metrics[i]]
		else:
			after_scores = pairwise_weighted_smape(path, after_metrics[i], id_array).flatten()
			scores[after_metrics[i]] = after_scores
		
		before_scores_arr += before_scores.tolist()
		after_scores_arr += after_scores.tolist()

	before_mean = np.mean(before_scores_arr)
	after_mean = np.mean(after_scores_arr)
	before_std = np.std(before_scores_arr)
	after_std = np.std(after_scores_arr)
	before_ci = 1.96 * before_std / np.sqrt(len(before_scores_arr))
	after_ci = 1.96 * after_std / np.sqrt(len(after_scores_arr))

	ax.scatter(before_mean, after_mean, s=20, c=color, marker=marker, alpha=0.5)
	ax.errorbar(before_mean, after_mean, xerr=before_ci, yerr=after_ci, c=color, alpha=0.5)

	ax.axline((0, 0), slope=1, c='red', linestyle='--')
	ax.axis('square')

	ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))


	
		
	# df = pd.DataFrame({
	# 	'Trick': tricks_arr,
	# 	'SMAPE': scores_arr
	# })

	# sns.barplot(x="Trick", y="SMAPE", data=df, ax=ax)

	# ax.set_title(ablation_name)

def plot_boxplot_ax(path, id_array, ablation_info, ablation_names, scores, ax, cmap_dict, is_ylabel):
	scores_arr = []
	tricks_arr = []
	for trick in ablation_info.keys():
		before_metrics = ablation_info[trick][False]
		after_metrics = ablation_info[trick][True]
		for i in range(len(before_metrics)):
			if before_metrics[i] in scores:
				before_scores = scores[before_metrics[i]]
			else:
				before_scores = pairwise_weighted_smape(path, before_metrics[i], id_array).flatten()
				scores[before_metrics[i]] = before_scores
			if after_metrics[i] in scores:
				after_scores = scores[after_metrics[i]]
			else:
				after_scores = pairwise_weighted_smape(path, after_metrics[i], id_array).flatten()
				scores[after_metrics[i]] = after_scores
			scores_arr += ((before_scores - after_scores) / (0.01 * np.maximum(before_scores, after_scores))).tolist()
			tricks_arr += [ablation_names[trick]] * len(before_scores)

		
			
	
	df = pd.DataFrame({
		'Trick': tricks_arr,
		'SMAPE Diff': scores_arr
	})

	## set y order by SMAPE Diff mean score
	trick_arr = np.unique(tricks_arr)
	mean_arr  = []
	for trick in trick_arr:
		mean_arr.append(np.mean(df[df['Trick'] == trick]['SMAPE Diff']))


	# trick_arr = trick_arr[np.argsort(mean_arr)]
	trick_arr = trick_arr[np.argsort([len(trick) for trick in trick_arr])]
	## swap the last and second last of trick_arr
	trick_arr[-1], trick_arr[-2] = trick_arr[-2], trick_arr[-1]
	trick_arr[1], trick_arr[2] = trick_arr[2], trick_arr[1]
	trick_arr[3], trick_arr[4] = trick_arr[4], trick_arr[3]

	cmap = [cmap_dict[trick] for trick in trick_arr]


	sns.barplot(x="Trick", y="SMAPE Diff", data=df, ax=ax, palette=cmap, order=trick_arr)

	if is_ylabel:
		ax.set_ylabel('SMAPE Decrement (%)')
	else:
		ax.set_ylabel('')
	

	
def plot_barchart_ax(path, metrics, id_array, xlabel, ax, cmap, is_log, is_first, ylim):
	metrics_array = []
	scores_array = []
	for metric in metrics:
		scores = pairwise_weighted_smape(path, metric, id_array).flatten()
		metrics_array += [metric] * scores.shape[0]
		scores_array += scores.tolist()

	df = pd.DataFrame({'measure': metrics_array, 'SMAPE': scores_array})


	sns.set_style("whitegrid")
	sns.barplot(x="measure", y="SMAPE", data=df, ax=ax, palette=cmap)

	if is_first:
		ax.set_ylabel("SMAPE")
	else:
		ax.set_ylabel("")
		ax.set_yticklabels([])
	
	ax.set_ylim(ylim[0], ylim[1])

	ax.set_xticklabels(
		[" "] * len(metrics),
	)
	ax.set_xlabel(xlabel)

	if is_log:
		ax.set_yscale("log")

		

		

def plot_heatmap(path, test, metric, scores, id_array):

	plt.figure(figsize = (12, 10))
	ax = sns.heatmap(
		scores, cmap="Blues", 
		vmin=0, vmax=1, 
		xticklabels=id_array, yticklabels=id_array,
		annot=True, fmt=".2f", 
		annot_kws={"size": 11,}
	)

	for t in ax.texts:
		if float(t.get_text()) < 0:
			t.set_text("< 0")

	fig = ax.get_figure()
	check_and_make(f"{path}/{metric}")
	fig.savefig(f"{path}/{metric}/{test}.png")
	plt.clf()
	

def plot_heatmap_ax(scores, id_array, ax):
	sns.heatmap(
		scores, cmap="Blues", 
		vmin=0, vmax=1, 
		xticklabels=id_array, yticklabels=id_array,
		annot=False, ax=ax, cbar=False
	)
	