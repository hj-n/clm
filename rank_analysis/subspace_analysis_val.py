import numpy as np
from data.reader import * 
from scorer.measures_scorer import *
from scorer.clustering_scorer import *
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

DATASET_LIST = np.load("./results/dataset_list.npy").tolist()
MAX_DIM = 15
MAX_SIZE = 5000


scorers_dict = {
	"hdbscan": ["HDBSCAN", hdbscan_scorer],
	"dbscan": ["DBSCAN", dbscan_scorer],
	"kmeans": ["K-Means", kmeans_scorer],
	"kmedoid": ["K-Medoid", kmedioid_scorer],
	"xmeans": ["X-Means", xmeans_scorer],
	"birch": ["Birch", birch_scorer],
	"agglo_complete": ["Agglomerative (Complete)", agglo_complete_scorer],
	"agglo_average": ["Agglomerative (Average)", agglo_average_scorer],
	"agglo_single": ["Agglomerative (Single)", agglo_single_scorer]
}

MAX_SUBSPACE = 50


ext_measure = "ami"
ext_measure_scorer = ami_scorer


NEW_DATASET_LIST = []
## check the dimensionality of the datasets
for dataset in DATASET_LIST:
	data, labels = read_dataset_by_path(f"./data/compressed/{dataset}/")
	if data.shape[1] <= MAX_DIM and data.shape[0] <= MAX_SIZE:
		NEW_DATASET_LIST.append(dataset)



for idx,dataset in enumerate(NEW_DATASET_LIST):
	print(f"Running {dataset} ({idx+1}/{len(NEW_DATASET_LIST)})")
	data, labels = read_dataset_by_path(f"./data/compressed/{dataset}/")
	unique_labels = np.unique(labels)
	label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
	labels_new= np.array([label_map[old_label] for old_label in labels], dtype=np.int32)

	## generate 50 random subspaces for each dataset including the worst and best scores
	best_subspace = np.load(f"./application_subspace/subspaces/{dataset}_ch_btw_best_subspace.npy")
	worst_subspace = np.load(f"./application_subspace/subspaces/{dataset}_ch_btw_worst_subspace.npy")


	subspaces = []
	subspaces.append("".join(best_subspace.astype(int).astype(str).tolist()))
	subspaces.append("".join(worst_subspace.astype(int).astype(str).tolist()))

	for i in range(MAX_SUBSPACE):
		subspace = np.random.randint(2, size=data.shape[1]).astype(bool)
		## check whether the subspace is all consists of false
		while np.sum(subspace.astype(int)) == 0:
			subspace = np.random.randint(2, size=data.shape[1]).astype(bool)
		subspaces.append("".join(subspace.astype(int).astype(str).tolist()))
	
	## run the scorers

	scorers_result_dict = {
		"hdbscan": [],
		"dbscan": [],
		"kmeans": [],
		"kmedoid": [],
		"xmeans": [],
		"birch": [],
		"agglo_complete": [],
		"agglo_average": [],
		"agglo_single": []
	}

	scorers_times_dict = {
		"hdbscan": [],
		"dbscan": [],
		"kmeans": [],
		"kmedoid": [],
		"xmeans": [],
		"birch": [],
		"agglo_complete": [],
		"agglo_average": [],
		"agglo_single": []
	}

	if os.path.exists(f"./application_subspace/ground_truth_50/{dataset}_subspaces.npy"):
		continue

	for subspace in tqdm(subspaces):
		subspace_bool = np.array(list(map(int, list(subspace)))).astype(bool)
		data_sub = data[:, subspace_bool]
		data_sub = np.ascontiguousarray(data_sub)
		for scorer_key in scorers_dict.keys():
			start = time.time() 
			scorer_name, scorer_func = scorers_dict[scorer_key]
			score = scorer_func(data_sub, labels_new, ext_measure_scorer)
			scorers_result_dict[scorer_key].append(score)
			end = time.time()
			scorers_times_dict[scorer_key].append(end-start)
	
	## save the results
	for scorer_key in scorers_dict.keys():
		np.save(f"./application_subspace/ground_truth_50/{dataset}_{scorer_key}_scores.npy", scorers_result_dict[scorer_key])
		np.save(f"./application_subspace/ground_truth_50/{dataset}_{scorer_key}_times.npy", scorers_times_dict[scorer_key])
	
	## save the subspaces
	np.save(f"./application_subspace/ground_truth_50/{dataset}_subspaces.npy", subspaces)
