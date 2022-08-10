# Sanity Check for External Clustering Validation Benchmarks using Internal Validation Measures - *Codes for Reproduction*

This repository contains the source codes for reproducing the results in the paper "Sanity Check for External Clustering Validation Benchmarks using Internal Validation Measures". Please follow the below instructions to run the experiments. 

### Dependencies & Envrionment

The code in this repository is mainly written in python. The list of dependencies is as follows:

- `numpy`
- `pandas`
- `tqdm`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `hdbscan`
- `scickit-learn-extra`
- `pyclustering`
- `bayesian-optimization`

For an easy installation of the dependencies, we provided a environment file `clm_env.yml`. The following command will automatically install all the dependencies.

```sh
conda env create -f clm_env.yml
conda activate clmenv
```

### Experiments

The experiments in our study are consists

### Ablation Study 

1. Run the `R` code `_RUN_ME_FOR_NEURIPS2022.R` code within directory `./ablation/data/data_generator` to generate 1,000 base datasets.
2. Go to the directory `./ablation/` and run the python scripts by using following commands:
  - *Shift invariance test*:
    - `python3 ablation.py -t shift -m all -f all`
    - the scores and plots will be stored in `./ablation/results_shift/scores/` and `./ablation/results_shift/plots/`, respectively.
  - *Data Cardinality test*:
    - `python3 ablation.py -t card -m all -f all`
    - the results and plots will be stored in `./ablation/results_card/scores/` and `./ablation/results_card/plots/`, respectively.
  - There are more obtions in `ablation.py`. Check the options with `python3 ablation.py -h` command.
3. After running `ablation.py`, run the following script to earn the bar chart summarizing the results of each test.
  - `python3 summary.py -t shift`
  - `python3 summary.py -t card`
 
 ### Between-Dataset Rank Correlation Analysis & Time Analysis
 
 1. Go to the directory `/rank_analysis/` and run the python scripts by using following commands:
   - `python3 measures_run.py -m all -t`
   - `python3 clusterings_run.py -m all -t`
   - Note that...
     - `-t` flag turns on the time analysis.
     - `measures_run.py` and `clusterings_run.py` might run more than a day, due to the running time of classifiers and clustering algorithms.
     - refer to the options by using `-h` flag.
 2. To compute the correlation between ground truth clusterings and estimated clusterings, run...
   - `python3 correlation.py -e ami`

	
