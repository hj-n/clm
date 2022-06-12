
from tqdm import tqdm

import argparse
import numpy as np
import os, json
import time 
from scorer.measures_scorer import *
from data.reader import *


DATASET_LIST = os.listdir("./data/compressed/")
DATASET_LIST.remove(".gitignore")

scorers_dict = {
	"ch"    : ["calinski_harabasz", calinski_scorer],
	"ch_btw": ["calinski_harabasz_btw", calinski_btw_scorer],
	"dunn"  : ["dunn", dunn_scorer],
	"db"    : ["davies_bouldin", davies_bouldin_scorer],
	"ii"    : ["i_index", i_index_scorer],
	"svm"   : ["support vector machine", svm_scorer],
	"knn"   : ["k-nearest neighbors", knn_scorer],
	"nb"    : ["naive bayes", nb_scorer],
	"rf"    : ["random forest", rf_scorer],
	"lr"    : ["logistic regression", logreg_scorer],
	"lda"   : ["linear discriminant analysis", lda_scorer],
}

#### Argument handling
parser = argparse.ArgumentParser(description="Obtain the CLM scores of the datasets", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--measure", "-m", type=str, default="all",
help=f"""run the specified measure, or all of them if 'all'
supported measures: {list(scorers_dict.keys())}"""
)
parser.add_argument("--time", "-t", action="store_true", help="run time analysis")

args = parser.parse_args()
measure_args = args.measure
tims_args = args.time

names = []
scorers = []
abbs = []
if measure_args == "all":
	for name, scorer in scorers_dict.items():
		names.append(scorer[0])
		scorers.append(scorer[1])
		abbs.append(name)
else:
	names.append(scorers_dict[measure_args][0])
	scorers.append(scorers_dict[measure_args][1])
	abbs.append(measure_args)


def run_measure(measure_scorer, measure_scorer_name, measure_abbreviation):
	scores = []
	times = []
	print("Running " + measure_scorer_name + " for datasets...")
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

		start = time.time()
		score = measure_scorer(data, labels_new)
		end = time.time()
		times.append(end - start)

		scores.append(score)

	print("finished...saving file...")
	with open(f"./results/measures/{measure_abbreviation}_score.json", "w") as f:
		json.dump(scores, f)
	if tims_args:
		with open(f"./results/measures/{measure_abbreviation}_time.json", "w") as f:
			json.dump(times, f)
	print("finished!!")

for i, name in enumerate(names):
	run_measure(scorers[i], name, abbs[i])