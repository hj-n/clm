import pandas as pd
import numpy as np

from tqdm import tqdm
import os

import sys
sys.path.append('../')

from measures import calinski_harabasz as ch
from measures import silhouette as sil
from measures import dunn 
from measures import xie_beni as xb
from measures import i_index as ii
from measures import davies_bouldin as db



def read_csv(file_name):
	df = pd.read_csv(file_name)
	return df

def get_num_dict():
	"""
	return the dictionary that maps the scatterplot key to the number of points
	"""
	## read all file names in the directory "./data/NEW_X_Y_LABELS_update/"
	file_names = os.listdir("./data/")
	file_names.remove(".gitignore")
	file_names.remove("NEW_X_Y_NOISE_LABELS_N10000_PART1.zip")
	file_names.remove("NEW_X_Y_NOISE_LABELS_N10000_PART2.zip")
	file_names.remove("NEW_X_Y_NOISE_LABELS_N10000_PART3.zip")
	## trim end of file names to remove seed and csv info and eliminate duplicate
	file_names = [file_name[:-10] for file_name in file_names]
	file_names = list(set(file_names))
	## split file names into key and value
	key_and_num = [file_name.split("_noise498_num") for file_name in file_names]
	## convert key_and_num to the dictionary format
	num_dict = {}
	for i in range(len(key_and_num)):
		num_dict[int(key_and_num[i][0])] = int(key_and_num[i][1])
	return num_dict

def get_scatterplot(key, num, dim, size):
	"""
	return a scatterplot with key and num
	"""
	## read csv file 
	df = read_csv("./data/{}_noise498_num{}_seed0.csv".format(key, num))

	df_np = df.to_numpy()
	filter_arr = np.array(range(0, len(df_np)))
	np.random.shuffle(filter_arr)
	filter_arr = filter_arr < size

	X = df_np[filter_arr, :dim] 
	labels = df_np[filter_arr, -1]

	labels[labels == 1] = 0
	labels[labels == 2] = 1


	return X, labels.astype(np.int32)

def test_single_scatterplot(key, num, metric_name, dim, size):
	"""
	test a single scatterplot having key and num with metrics metric_name
	"""
	score = 0
	X, labels = get_scatterplot(key, num, dim, size)
	## Calinski-Harabasz
	if metric_name == "ch":
		return ch.calinski_harabasz(X, labels)
	elif metric_name == "ch_shift":
		return ch.calinski_harabasz_shift(X, labels)
	elif metric_name == "ch_range":
		return ch.calinski_harabasz_range(X, labels)
	elif metric_name == "ch_btw":
		return ch.calinski_harabasz_btw(X, labels)
	## Silhouette
	elif metric_name == "sil":
		return sil.silhouette(X, labels)
	elif metric_name == "sil_shift":
		return sil.silhouette_shift(X, labels)
	elif metric_name == "sil_btw":
		return sil.silhouette_btw(X, labels)
	## Dunn
	elif metric_name == "dunn":
		return dunn.dunn(X, labels)
	elif metric_name == "dunn_shift":
		return dunn.dunn_shift(X, labels)
	elif metric_name == "dunn_range":
		return dunn.dunn_range(X, labels)
	elif metric_name == "dunn_btw":
		return dunn.dunn_btw(X, labels)
	## I-index
	elif metric_name == "ii":
		return ii.i_index(X, labels)
	elif metric_name == "ii_shift":
		return ii.i_index_shift(X, labels)
	elif metric_name == "ii_range":
		return ii.i_index_range(X, labels)
	elif metric_name == "ii_btw":
		return ii.i_index_btw(X, labels)
	## Xie-Beni
	elif metric_name == "xb":
		return xb.xie_beni(X, labels)
	elif metric_name == "xb_shift":
		return xb.xie_beni_shift(X, labels)
	elif metric_name == "xb_range":
		return xb.xie_beni_range(X, labels)
	elif metric_name == "xb_btw":
		return xb.xie_beni_btw(X, labels)
	## Davies-Bouldin
	elif metric_name == "db":
		return db.davies_bouldin(X, labels)
	else:
		raise Exception("Invalid metric name")



def test_all_scatterplots(metric_name, dim, size):
	"""
	test all scatterplots with metrics metric_name
	return 2D array where each row corresponds to a scatterplot, 
	and the first and the second column corresponds to probMore and the score tested by test_single_scatterplot
	"""
	num_dict = get_num_dict()
	scores = []
	for i, key in enumerate(tqdm(num_dict.keys())):
		num = num_dict[key]

		dim_i = dim[i] if isinstance(dim, np.ndarray) else dim
		size_i = size[i] if isinstance(size, np.ndarray) else size

		score = test_single_scatterplot(key, num, metric_name, dim_i, size_i)
		scores.append(score)


	return np.array(scores)


### RUN ###
def run(metric_name, dim, point_num):
	scores = test_all_scatterplots(metric_name, dim, point_num)
	return scores



