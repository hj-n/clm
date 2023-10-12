# Adjusted Internal Validation Measures ($`IVM{}_A`$)

This repository contains the source codes of adjusted internal validation measures (adjusted IVMs), introduced by our paper "Measuring the Validity of Clustering Validation Dataset" (Under review). 

## What are adjusted IVMs?

Adjusted Internal Validation Measures (IVM$`{}_A`$) are introduced as a method to evaluate and compare Cluster-Label Matching (CLM) across various datasets. CLM refers to how well the class labels of a dataset align with actual data clusters, which is crucial for accurately validating unsupervised clustering techniques using benchmark datasets where class labels are utilized as ground-truth clusters. Traditional Internal Validation Measures (IVMs), such as the Calinski-Harabasz index or the Silhouette coefficient, are capable of comparing CLM over different labelings of the same dataset but are not designed to do so across different datasets. IVM$`{}_A`$s, on the other hand, are proposed to evaluate and compare CLM across datasets in a fast and reliable manner, taking a labeled dataset as input and returning a score that evaluates its level of CLM. Unlike standard IVM scores, those ofIVM$`{}_A`$ are comparable both across and within datasets.

The development and application of IVM$`{}_A`$s involve a structured approach and several key contributions. Four axioms are proposed that form the grounded basis of IVMAs, complementing existing within-dataset axioms and requiring IVM$`{}_A`$ to be invariant with the dimensionality, number of data points, and number of classes, and to share a common range so that their scores can be compared across different datasets. A procedure for adjusting an IVM into an IVM$`{}_A`$ is proposed, involving four technical processes designed to make IVM satisfy the across-dataset axioms while still fulfilling the within-dataset axioms. Six widely used IVMs (Calinski-Harabasz, Dunn Index, I Index, Xie-Beni Index, Davies-Bouldin Index, and Silhouette Coefficient) are generalized into IVM$`{}_A`$s using these processes.

### Dependencies & Envrionment

The code in this repository is mainly written in Python. The list of dependencies is as follows:

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

For an easy installation of the dependencies, we provided an environment file `clm_env.yml`. The following command will automatically install all the dependencies.

```sh
conda env create -f clm_env.yml
conda activate clmenv
```

### Supported Internal validation measures

We generalized six internal validation measures: Calinski-Harabasz ($CH$), Dunn Index ($DI$), I Index ($II$), Xie-Beni Index ($XB$), Davies-Bouldin Index ($DB$), and Silhouette Coefficient ($SC$). As a result, we obtained five adjusted internal validation measures (IVM$`{}_A`$s): $CH_A$, $DI_A$, $`\{II, XB\}_A`$, $DB_A$, and $SC_A$. Note that $II_A$ and $XB_A$ become identical after passing through our generalization processes. Please refer to the below API description to invoke IVM$`{}_A`$s.


### Contact

Please contact [hj@hcil.snu.ac.kr](mailto:hcil.snu.ac.kr) if there exists any issue executing the codes.
  
### Reference

TBA
 

	
