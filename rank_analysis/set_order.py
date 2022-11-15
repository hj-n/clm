import os 
import numpy as np

DATASET_LIST = os.listdir("./data/compressed/")
DATASET_LIST.remove(".gitignore")

print(DATASET_LIST)

np.save("./results/dataset_list.npy", np.array(DATASET_LIST))
