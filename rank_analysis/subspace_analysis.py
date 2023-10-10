import numpy as np
from data.reader import *
from scorer.measures_scorer import *
from tqdm import tqdm
import time

from sklearn.preprocessing import StandardScaler

DATASET_LIST = np.load("./results/dataset_list.npy").tolist()
# MAX_DIM = 15
# MAX_SIZE = 5000

MEASURE = "ch_btw"
SCORER  = calinski_btw_scorer

NEW_DATASET_LIST = DATASET_LIST 

## find the best subspace for each dataset that maximizes the MEASURE SCORE
times = []
clm_initials = []
clm_bests = []
best_weights = []
for idx, dataset in enumerate(NEW_DATASET_LIST):
	print(f"Running {MEASURE} for dataset {dataset} ({idx+1}/{len(NEW_DATASET_LIST)})")
	data, labels = read_dataset_by_path(f"./data/compressed/{dataset}/")
	data_max = np.max(data)
	data_min = np.min(data)
	data_norm = np.abs(data_max) if np.abs(data_max) > np.abs(data_min) else np.abs(data_min)
	data = data / data_norm
	unique_labels = np.unique(labels)
	label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
	labels_new= np.array([label_map[old_label] for old_label in labels], dtype=np.int32)
	best_score = -1
	worst_score = 1e9
	best_subspace = np.zeros(data.shape[1]).astype(bool)
	worst_subspace = np.zeros(data.shape[1]).astype(bool)

	clm_initial = SCORER(data, labels_new)

	start = time.time()

	for i in tqdm(range(1000)):
		## generate a random weight vector where each element is between 0 and 1 (uniform distribution)\
		weight = np.random.uniform(0, 1, data.shape[1])
		## apply the weight vector to the data
		data_weighted = data * weight
		score = SCORER(data_weighted, labels_new)
		if score > best_score:
			best_score = score
			best_subspace = weight
		if score < worst_score:
			worst_score = score
			worst_subspace = weight

	end = time.time()
	
	
	clm_best = SCORER(data * best_subspace, labels_new)

	times.append(end - start)
	clm_initials.append(clm_initial)
	clm_bests.append(clm_best)
	best_weights.append(best_subspace)


best_weights = np.array(best_weights)


	
np.save(f"./application_subspace/subspaces/{MEASURE}_clm_initials.npy", np.array(clm_initials))
np.save(f"./application_subspace/subspaces/{MEASURE}_clm_bests.npy", np.array(clm_bests))
np.save(f"./application_subspace/subspaces/{MEASURE}_times.npy", np.array(times))
np.save(f"./application_subspace/subspaces/{MEASURE}_best_weights.npy", np.array(best_weights))





