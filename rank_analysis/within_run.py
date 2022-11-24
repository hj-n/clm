from tqdm import tqdm 
import numpy as np
from scorer.measures_scorer import *
from data.reader import *
import copy
import os

DATASET_LIST = np.load("./results/dataset_list.npy")
import argparse

noise_level = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

parser = argparse.ArgumentParser(description="Obtain the CLM scores of the datasets by changing nosie level", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--measure", "-m", type=str, default="ch")

scorers_dict = {
	"ch"      : ["calinski_harabasz", calinski_scorer],
	"ch_btw"  : ["calinski_harabasz_btw", calinski_btw_scorer],
	"dunn"    : ["dunn", dunn_scorer],
	"dunn_btw": ["dunn_btw", dunn_btw_scorer],
	"db"      : ["davies_bouldin", davies_bouldin_scorer],
	"ii"      : ["i_index", i_index_scorer],
	"ii_btw"  : ["i_index_btw", i_index_btw_scorer],
	"sil"     : ["silhouette", silhouette_scorer],
	"sil_btw" : ["silhouette_btw", silhouette_btw_scorer],
	"xb"      : ["xie_beni", xie_beni_scorer],

}
args = parser.parse_args()
measure = args.measure

for dataset in tqdm(DATASET_LIST):
	data, labels = read_dataset_by_path(f"./data/compressed/{dataset}/")
	data = np.array(data)
	labels = np.array(labels)
	data_max = np.max(data)
	data_min = np.min(data)
	data_norm = np.abs(data_max) if np.abs(data_max) > np.abs(data_min) else np.abs(data_min)
	data = data / data_norm

	unique_labels = np.unique(labels)
	label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
	labels_new= np.array([label_map[old_label] for old_label in labels], dtype=np.int32)
	labels = labels_new

	scores = []
	for noise in noise_level:
		indices = np.random.choice(np.arange(len(labels)), int(len(labels) * noise), replace=False)
		labels_copy = copy.deepcopy(labels)
		labels_copy_subset = labels_copy[indices]
		np.random.shuffle(labels_copy_subset)
		labels_copy[indices] = labels_copy_subset
		score = scorers_dict[measure][1](data, labels_copy)
		scores.append(score)
	
	if os.path.exists(f"./within_results/scores/{measure}/") == False:
		os.makedirs(f"./within_results/scores/{measure}/")
	
	np.save(f"./within_results/scores/{measure}/{dataset}", scores)
	

	

