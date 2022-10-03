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

Out experiments are consists of two parts: ablation study (Section 5.1) and between-dataset rank correlation analysis (Section 5.2.). Two experiments can be reproduced by executing the code within directories `ablation` and `rank_analysis`. Please follow the below instructions to run the code. 

#### Ablation Study 

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
  - The resulting plots are also depicted in Figure 2 of the paper. 
 
 #### Between-Dataset Rank Correlation Analysis
 
 1. Go to the directory `/rank_analysis/` and run the python scripts by using following commands:
   - `python3 measures_run.py -m all -t`
   - `python3 clusterings_run.py -m all -t`
   - Note that...
     - `-t` flag turns on the time analysis.
     - `measures_run.py` and `clusterings_run.py` might run more than a day, due to the running time of classifiers and clustering algorithms.
     - refer to the options by using `-h` flag.
 2. To compute the correlation between ground truth clusterings and estimated clusterings, run...
   -  (for `ami`) `python3 correlation.py -e ami`
   -  (for `arand`) `python3 correlation.py -e arand`
   -  (for `vm`) `python3 correlation.py -e vm`
   -  (for `nmi`) `python3 correlation.py -e nmi`
   - The results produced by `correltaion.py` can be found in Table 1 of the main manuscript. 
   
 #### Time Analysis
 
After executing `measures_run.py` and `clusterings_run.py` for all measures and clustering algorithms (with ami)  with `-t` flag, enter the following commands to check the running time.
   - `python3 time.py`
   - The result of time analysis can be found in Appendix F.
 
 #### Summary
 
After executing `measures_run.py` and `clusterings_run.py` for all measures and clustering algorithms (with entire external measures), the summary statistics can be generated using the following command:
  - `python3 summary.py` 
  - The resulting csv file containing summary statistics is stored in [our dataset repository](https://github.com/hj-n/labeled-datasets) and can be also found in [the dataset website](https://hyeonword.com/clm-datasets/).

### Contact

Please contact [hj@hcil.snu.ac.kr](mailto:hcil.snu.ac.kr) if there exists any issue executing the codes.
  
### Reference

TBA
 

	
