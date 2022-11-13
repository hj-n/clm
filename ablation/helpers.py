import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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

	data_1 = read_json(path_1)[:]
	data_2 = read_json(path_2)[:]
	return np.mean(np.abs(data_1 - data_2) / (np.abs(data_1) + np.abs(data_2)))

def pairwise_smape(path, metric, id_array):
	scores = np.zeros((id_array.shape[0], id_array.shape[0]))
	for i, id1 in enumerate(id_array):
		for j, id2 in enumerate(id_array):
			scores[i, j] = smape(path, metric, id1, id2)

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
	
	