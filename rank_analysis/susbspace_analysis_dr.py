import numpy as np

import os
from data.reader import *

import umap

from tqdm import tqdm

import matplotlib.pyplot as plt

file_list = os.listdir("./application_subspace/subspaces/")
file_list = [file for file in file_list if file.endswith("best_subspace.npy")]
dataset_list = [file[:-len("_ch_btw_best_subspace.npy")] for file in file_list]

for dataset in tqdm(dataset_list):
  data, labels = read_dataset_by_path(f"./data/compressed/{dataset}/")
  best_subspace = np.load(f"./application_subspace/subspaces/{dataset}_ch_btw_best_subspace.npy")
  
  unique_labels = np.unique(labels)
  label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
  labels_new= np.array([label_map[old_label] for old_label in labels], dtype=np.int32)
	


  data_sub = data[:, best_subspace]
  
  umap_reducer = umap.UMAP() 

  data_embedding = umap_reducer.fit_transform(data)
  data_sub_embedding = umap_reducer.fit_transform(data_sub)

  plt.scatter(data_embedding[:,0], data_embedding[:,1], c=labels_new, s=5)
  plt.savefig(f"./application_subspace/dr/{dataset}_embedding.png")
  
  plt.clf()
  
  plt.scatter(data_sub_embedding[:,0], data_sub_embedding[:,1], c=labels_new, s=5)
  plt.savefig(f"./application_subspace/dr/{dataset}_subspace_embedding.png")
  
  plt.clf()
  
  