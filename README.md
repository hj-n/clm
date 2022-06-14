# Class-Label Matching

This repository contains the source codes for reproducing the experiments in the paper "Sanity Check for External Clustering Validation Benchmarks using Internal Validation Measures". Please follow the below instructions to run the experiments. 

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
 2. (Rank Correlation Analysis)
 3. (Time Analysis)

	
