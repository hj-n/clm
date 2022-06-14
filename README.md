# Class-Label Matching

This repository contains the source codes for reproducing the experiments in the paper "Sanity Check for External Clustering Validation Benchmarks using Internal Validation Measures". Please follow the below instructions to run the experiments. 

### Ablation Study 

#### Step 1: Dataset Generation

1. Run the `R` code `_RUN_ME_FOR_NEURIPS2022.R` code within directory `./ablation/data/data_generator` to generate 1,000 base datasets.
2. Run the python scripts by using following commands:
  - Shift invariance test:
	  - ```sh
		python3 ablation.py -t shift -m all -f all
		```