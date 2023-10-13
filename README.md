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

### API

Every function (both standard and adjusted IVMs) have same interface. The following is the description of the interface.

```python
def function_name(
	data,
	labels
)
```

`data` is a numpy array of shape `(n_samples, n_features)`, where `n_samples` is the number of data points and `n_features` is the number of features. `labels` is a numpy array of shape `(n_samples,)`, where `n_samples` is the number of data points. `labels` contains the class labels of the data points.

Note that adjusted IVMs additionally have a hyperparameter $k$, which controls the skewness of the data scores (does not alter the ranking of the scores). However, we highly recommend not to change the default value of $k$, as it is already optimized to be an optimal.

The list of supported functions are as follows:

#### Internal Validation Measures
- `calinski_harabasz`: Calinski-Harabasz index
- `dunn_index`: Dunn index
- `i_index`: I index
- `xie_beni_index`: Xie-Beni index
- `davies_bouldin_index`: Davies-Bouldin index
- `silhouette`: Silhouette coefficient

#### Adjusted Internal Validation Measures
- `calinski_harabasz_adjusted`: Adjusted Calinski-Harabasz index
- `dunn_index_adjusted`: Adjusted Dunn index
- `i_index_adjusted`: Adjusted I index
- `xie_beni_index_adjusted`: Adjusted Xie-Beni index
- `davies_bouldin_index_adjusted`: Adjusted Davies-Bouldin index
- `silhouette_adjusted`: Adjusted Silhouette coefficient

You can simply invoke the function by substuting `function_name` with the name of the function you want to use. For example, if you want to use the Calinski-Harabasz index, you can invoke the function as follows:

```python
from measures import calinski_harabasz as ch
from sklearn.datasets import load_iris

data, labels = load_iris(return_X_y=True)

### standard IVM
ch_score = ch.calinski_harabasz(data, labels)

### adjusted IVM
ch_a_score = ch.calinski_harabasz_adjusted(data, labels)

```


### Contact

Please contact [hj@hcil.snu.ac.kr](mailto:hcil.snu.ac.kr) if there exists any issue executing the codes.
  
### Reference

TBA
 

	
