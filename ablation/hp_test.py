import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import metric_run as mer
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Hyperparameter Setting test", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--measure", "-m", type=str, default="all", help='''set the measure used for the test''')
parser.add_argument("--ablation", "-a", type=str, default="all", help='''set the ablation type used for the test''')

args = parser.parse_args()
measure = args.measure
ablation = args.ablation

## dataset loading
path = "./smalldata/"
files = os.listdir(path)
if ".gitignore" in files:
	files.remove(".gitignore")

judgement_data = pd.read_csv("./data_metadata/1000_2gaussians_proba_judgment_ClustMe_EXP1.csv")
judgement_dict = {}
probMore_data = judgement_data["probMore"].to_numpy()
seed_data = judgement_data["XYposCSVfilename"].to_numpy()
for i in range(len(probMore_data)):
	judgement_dict[seed_data[i]] = probMore_data[i]


binning = 10
binspace = np.linspace(0, 1, binning+1)[:binning]
binweights = np.zeros(binspace.shape, dtype=np.int32)
bininterval = 1.0 / binning

for i in range(len(judgement_data)):
	idx = int(probMore_data[i] // bininterval)
	if idx == binning:
		idx = binning - 1
	binweights[idx] += 1

binweights = 1 / binweights
binweights = binweights / np.sum(binweights)

seed_to_weights = {}
for i in range(len(seed_data)):
	idx = int(probMore_data[i] // bininterval)
	if idx == binning:
		idx = binning - 1
	seed_to_weights[seed_data[i]] = binweights[idx]



#### dataset
dataset_arr   = []
seed_arr      = []
label_arr     = []
judgement_arr = []
weights_arr   = []

print("reading data...")
for file in tqdm(files):
	dataset_full = pd.read_csv(path + file, header=None).to_numpy()
	dataset = dataset_full[1:, 0:2]
	label = dataset_full[1:, -1]
	seed = file.split("_")[0]
	dataset_arr.append(np.array(dataset, dtype=np.float64))
	seed_arr.append(int(seed))
	labels = np.array(label, dtype=np.float64).astype(int)
	labels = np.array([0 if x == 1 else 1 for x in labels])
	label_arr.append(labels)
	judgement_arr.append(judgement_dict[int(seed)])
	weights_arr.append(seed_to_weights[int(seed)])


### change 1 => 0, 2 => 1


### measuring score 

measure_name = f"{measure}_{ablation}"
pbounds = { 'k': (0,100)}


def run_test(k):
	score_arr = []
	for i in range(len(dataset_arr)):
		score = mer.metric_run_single_k(measure_name, dataset_arr[i], label_arr[i], k)
		score_arr.append(score)
	
	
	return r2_score(judgement_arr, score_arr, sample_weight=weights_arr)
	

# bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.5)


optimizer = BayesianOptimization(
	f=run_test,
	pbounds=pbounds,
)

optimizer.maximize(
	init_points=20,
	n_iter=50,
)

print("k value:", optimizer.max['params']['k'])

k= optimizer.max['params']['k']


real_score_arr = []
for i in range(len(dataset_arr)):
	score = mer.metric_run_single_k(measure_name, dataset_arr[i], label_arr[i], k)
	real_score_arr.append(score)

## savefig
plt.scatter(judgement_arr, real_score_arr)
plt.xlabel("human judgement")
plt.ylabel(f"{measure_name} score (k={k}")

plt.savefig(f"./hp_plot/{measure_name}.png")

