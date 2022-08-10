
from tqdm import tqdm

import argparse
import numpy as np
import os, json
import time

from scorer.clustering_scorer import *
from data.reader import *

DATASET_LIST = os.listdir("./data/compressed/")
DATASET_LIST.remove(".gitignore")



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



ext_measures_dict = {
	"ami": ["Adjusted Mutual Information", ami_scorer],
	"arand": ["Adjusted Rand Index", arand_scorer],
	"vm": ["V-Measure", vm_scorer],
	"nmi": ["Normalized Mutual Information", nmi_scorer]
}

parser = argparse.ArgumentParser(description="Obtain the ground truth CLM scores of the datasets using clustering algorithms", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--clustering", "-c", type=str, default="all",
help=f"""run the specified clustering algorithm, or all of them if 'all'
supported clustering algorithms: {list(scorers_dict.keys())}"""
)
parser.add_argument("--external-measure", "-e", type=str, default="ami", 
help=f"""select the external measure to use
supported external measures: {list(ext_measures_dict.keys())}"""
)
parser.add_argument("--time", "-t", action="store_true", help="run time analysis")

args = parser.parse_args()

clustering_names = []
clustering_scores = []
clustering_abbs = []
if args.clustering == "all":
	for name, scorer in scorers_dict.items():
		clustering_names.append(scorer[0])
		clustering_scores.append(scorer[1])
		clustering_abbs.append(name)
else:
	clustering_names.append(scorers_dict[args.clustering][0])
	clustering_scores.append(scorers_dict[args.clustering][1])
	clustering_abbs.append(args.clustering)



ext_measure_name = ext_measures_dict[args.external_measure][0]
ext_measure_scorer = ext_measures_dict[args.external_measure][1]
ext_measure_abb = args.external_measure


def run(
	clustering_scorer, clustering_name, clustering_abb,
	ext_measure_scorer, ext_measure_name, ext_measure_abb
):
	scores = []
	times = []
	print("Running " + clustering_name + " with " + ext_measure_name + "...")
	for dataset in tqdm(DATASET_LIST):
		data, labels = read_dataset_by_path(f"./data/compressed/{dataset}/")
		data_max = np.max(data)
		data_min = np.min(data)
		data_norm = np.abs(data_max) if np.abs(data_max) > np.abs(data_min) else np.abs(data_min)
		
		start = time.time()
		score = clustering_scorer(data / data_norm, labels, ext_measure_scorer)
		end = time.time()
		scores.append(score)
		times.append(end - start)

	
	print("saving files...")
	with open(f"./results/clusterings/{clustering_abb}_{ext_measure_abb}_score.json", "w") as file:
		json.dump(scores, file)
	if args.time:
		with open(f"./results/clusterings/{clustering_abb}_{ext_measure_abb}_time.json", "w") as file:
			json.dump(times, file)
	
	print("finished!!")



for i, clustering_name in enumerate(clustering_names):
	run(
		clustering_scores[i], clustering_name, clustering_abbs[i],
		ext_measure_scorer, ext_measure_name, ext_measure_abb
	)