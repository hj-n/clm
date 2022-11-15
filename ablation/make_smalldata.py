import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_and_save_smallsize():
	file_names = os.listdir("./smalldata/")
	file_names.remove(".gitignore")


	for file_name in file_names:
		data = pd.read_csv("./smalldata/{}".format(file_name))
		# data = data.sample(n=1000)
		## count the labels
		labels = data.to_numpy()[:, -1]
		## count 0, 1 in labels
		count_0 = np.sum(labels == 1)
		count_1 = np.sum(labels == 2)
		# print(labels)
		print("count_0: {}, count_1: {}".format(count_0, count_1))
		# data.to_csv("./smalldata/{}".format(file_name), index=False)


create_and_save_smallsize()