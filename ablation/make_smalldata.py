import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_and_save_smallsize():
	file_names = os.listdir("./smalldata_2/")
	file_names.remove(".gitignore")


	for file_name in tqdm(file_names):
		data = pd.read_csv("./smalldata_2/{}".format(file_name))
		new_data = pd.DataFrame({
			"x": data["x"],
			"y": data["y"],
			"label": data["label"]
		})

		new_data.to_csv("./smalldata/{}".format(file_name), index=False)


create_and_save_smallsize()