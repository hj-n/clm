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

	return np.mean(weight[:-1] * (np.abs(data_1 - data_2) / (np.abs(data_1) + np.abs(data_2))))

	# smape_nominator = 0
	# smape_denominator = 0
	# for i in range(data_1.shape[0]):
	# 	smape_nominator += np.abs(data_1[i] - data_2[i]) * weight[i]
	# 	smape_denominator += np.abs(data_1[i]) + np.abs(data_2[i])

	# return smape_nominator / smape_denominator



def pairwise_weighted_smape(path, metric, id_array):
	## compute weight
	human_judgement = pd.read_csv("./data_metadata/1000_2gaussians_proba_judgment_ClustMe_EXP1.csv")
	human_judgement = human_judgement["probSingle"].to_numpy()
	fs = np.zeros(11)
	for hj in human_judgement:
		if int(hj * 10) == 10:
			fs[9] += 1
		else:
			fs[int(hj * 10)] += 1
	fs[10] = 1 ## temproal (to prevent divide-by-zero)
	ws = 1 / fs
	weight = ws / (np.sum(ws) - 1)
	weight[10] = weight[9]
	human_judgement_weight = [weight[int(idx)] for idx in human_judgement]


	scores = np.zeros((id_array.shape[0], id_array.shape[0]))
	for i, id1 in enumerate(id_array):
		for j, id2 in enumerate(id_array):
			scores[i, j] = weighted_smape(path, metric, human_judgement_weight, id1, id2)

	return scores

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
		scores = pairwise_smape(path, metric, id_array)
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

def plot_barchart_ax(path, metrics, id_array, xlabel, ax, cmap, is_log, is_first, ylim):
	score_arry = []
	metrics_array = []
	scores_array = []
	for metric in metrics:
		scores = pairwise_smape(path, metric, id_array).flatten()
		metrics_array += [metric] * scores.shape[0]
		scores_array += scores.tolist()
		# scores = pairwise_rsq(path, metric, id_array)
		# mean_scores = np.sum(scores) / ((scores.shape[0] - 1) * scores.shape[0])
		# score_arry.append(mean_scores)


	df = pd.DataFrame({'measure': metrics_array, 'SMAPE': scores_array})


	sns.set_style("whitegrid")
	# sns.boxplot(x="measure", y="SMAPE", data=df, ax=ax, palette=cmap)
	# sns.pointplot(x="measure", y="SMAPE", data=df, ax=ax, palette=cmap, join=False)
	sns.barplot(x="measure", y="SMAPE", data=df, ax=ax, palette=cmap)

	if is_first:
		ax.set_ylabel("SMAPE")
	else:
		ax.set_ylabel("")
		## make ytick invisible
		ax.set_yticklabels([])
	
	ax.set_ylim(ylim[0], ylim[1])

	ax.set_xticklabels(
		[" "] * len(metrics),
	)
	# if xlabel != "Calinski-Harabasz":
	# 	ax.set_ylabel("")
	ax.set_xlabel(xlabel)

	# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

	

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
	