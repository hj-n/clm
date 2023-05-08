import numpy as np
from data.reader import *
from scorer.measures_scorer import *
from tqdm import tqdm
import time

DATASET_LIST = np.load("./results/dataset_list.npy").tolist()
MAX_DIM = 15
MAX_SIZE = 5000

MEASURE = "db_btw"
SCORER  = xie_beni_btw_scorer

NEW_DATASET_LIST = []
## check the dimensionality of the datasets
for dataset in DATASET_LIST:
	data, labels = read_dataset_by_path(f"./data/compressed/{dataset}/")
	if data.shape[1] <= MAX_DIM and data.shape[0] <= MAX_SIZE:
		NEW_DATASET_LIST.append(dataset)



## find the best subspace for each dataset that maximizes the MEASURE SCORE
for idx, dataset in enumerate(NEW_DATASET_LIST):
	print(f"Running {MEASURE} for dataset {dataset} ({idx+1}/{len(NEW_DATASET_LIST)})")
	data, labels = read_dataset_by_path(f"./data/compressed/{dataset}/")
	unique_labels = np.unique(labels)
	label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
	labels_new= np.array([label_map[old_label] for old_label in labels], dtype=np.int32)
	best_score = -1
	worst_score = 1e9
	best_subspace = np.zeros(data.shape[1]).astype(bool)
	worst_subspace = np.zeros(data.shape[1]).astype(bool)
	
	times_list = []
	for i in tqdm(range(1, 2 ** data.shape[1])):
		start = time.time()
		subspace = np.array(list(map(int, list(bin(i)[2:].zfill(data.shape[1]))))).astype(bool)
		data_sub = data[:, subspace]
		score = SCORER(data_sub, labels_new)
		if score > best_score:
			best_score = score
			best_subspace = subspace
		if score < worst_score:
			worst_score = score
			worst_subspace = subspace
		end = time.time()
		times_list.append(end - start)


	np.save(f"./application_subspace/subspaces/{dataset}_{MEASURE}_best_subspace.npy", best_subspace)
	np.save(f"./application_subspace/subspaces/{dataset}_{MEASURE}_worst_subspace.npy", worst_subspace)
	np.save(f"./application_subspace/subspaces/{dataset}_{MEASURE}_times.npy", np.array(times_list))

