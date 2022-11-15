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

def smape(path, metric, id1, id2):
	path_1 = f"./{path}/{metric}/{id1}.json"
	path_2 = f"{path}/{metric}/{id2}.json"

	data_1 = np.array(read_json(path_1)[:])
	data_2 = np.array(read_json(path_2)[:])
	# print(data_1, data_2)
	## make elemnt smaller than 0 to 0
	return np.mean(np.abs(data_1 - data_2) / (np.abs(data_1) + np.abs(data_2)))
	# data_1[data_1 < 0] = 0
	# data_2[data_2 < 0] = 0
	# smape_nominator = 0
	# smape_denominator = 0
	# for i in range(data_1.shape[0]):
	# 	smape_nominator += np.abs(data_1[i] - data_2[i])
	# 	smape_denominator += data_1[i] + data_2[i]


	
	
	# return smape_nominator / smape_denominator
	# smape_arr = []
	# for i in range(data_1.shape[0]):
	# 	if data_1[i] == 0 and data_2[i] == 0:
	# 		smape_arr.append(0)
	# 	else:
	# 		smape_arr.append(abs(data_1[i] - data_2[i]) / (abs(data_1[i]) + abs(data_2[i])))

	# return np.mean(smape_arr)

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
		scores = pairwise_rsq(path, metric, id_array)
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

def plot_barchart_ax(path, metrics, id_array, xlabel, ax, cmap, is_log):
	score_arry = []
	for metric in metrics:
		scores = pairwise_smape(path, metric, id_array)
		# scores = pairwise_rsq(path, metric, id_array)
		mean_scores = np.sum(scores) / ((scores.shape[0] - 1) * scores.shape[0])
		score_arry.append(mean_scores)


	df = pd.DataFrame({'measure': metrics, 'SMAPE': score_arry})


	sns.set_style("whitegrid")
	sns.barplot(x="measure", y="SMAPE", data=df, ax=ax, palette=cmap)
	ax.set_xticklabels(
		[" "] * len(metrics),
	)
	if xlabel != "Calinski-Harabasz":
		ax.set_ylabel("")
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
	