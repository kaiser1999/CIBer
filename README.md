# Repository for CIBer

Paper: A New Simple Effective InsurTech Tool: Comonotone-Independence Bayes classifier (CIBer)
Author: Yongzhao CHEN, Ka Chun CHEUNG, Nok Sang FAN, James Cheng PENG, and Sheung Chi Phillip YAM

# User Guide for CIBer

This is the user guide for Comonotone-Independence Bayesian Classifier (CIBer). CIBer is a supervised learning model which deals with multi-class classification tasks. The document consists of four parts: the requirements for the input data, important functions in comonotonic.py, important functions in utils.py, a demo.

 

## Data Requirements

CIBer deals with multi-class classification tasks with numerical or discrete (but should be ordered) input variables. Before passing the data into the model, please make sure to do some proper preprocessing beforehand including outlier removal, scaling, normalizing. If your dataset contains categorical data like gender, nationality, occupation, which are not ordered, please refer to the function $\textbf{joint_encode}$ in the utils.py. 



## comonotonic.py

### init

$\textbf{x_train}$: input variables in the training set, which should be a numpy 2d array where each row represents an instance and each column represents an input variable.

$\textbf{y_train}$: labels of the training set, which should be $0,1,2,...$

$\textbf{discrete_feature_val}$: a dictionary, whose key is the index of each discrete variable (including ordered and unordered), value is the cardinality of the sample space for this variable. 

$\textbf{cont_col}$: a list, containing the indices of the continuous variables

$\textbf{categorical}$: a list, containing the indices of the categorical variables (no ordering)

$\textbf{min_corr}$: a number between $0,1$ which specifies the threshold of correlation when determining the comonotonic relationship

$\textbf{cate_clusters}$: a list of list, specifing the clusters of categorical variables. This can be achieved by calling the function $\textbf{joint_encode}$, which will be discussed later. 

$\textbf{corrtype}$: can be set to "pearson", "spearman", "kendall", "mutual_info". Four measurements to correlation. The default is "pearson".

$\textbf{discrete_method}$: can be set to "auto", "mdlp", "custom". In the third case, you need to pass in an allocation book.

$\textbf{allocation_book}$: a dictionary, whose key is the variable index and value is the number of bins you wish to discretize. Equal length bins will be provided.

### discretize

This function discretizes the continuous variables using the method specified by $\textbf{discrete_method}$.

### construct_feature_val

This function is called after discretization. To merge the feature value for continuous and discrete variables. 

### get_prior_prob

This function gets the probability for each class.

### get_cond_prob

This function returns the conditional marginal probability of each variable. It returns a dictionary {variable index: {class index: {value: conditional marginal probability} } }.

### complete_posterior_prob

 This function calls get_cond_prob and compute the conditional probability for every input variable.

### get_prob_interval

Now that the conditional marginal probabilities are got, this function returns the corresponding intervals for each variable conditioning on the labels. 

### clustering

This function uses the clustering algorithm to find a partition for the variables heuristically.

### get_como_prob_interval

Returns a dictionary for the intervals. {variable index: {class index: {variable value: [infimum, supremum]}}}

### interval_intersection

Get the intersection of intervals

### get_prob_dist_single

Get the probability distribution over all classes for a single sample

### predict_proba

Returns the probability distribution over all classes



## utils.py

### simple_encode

This function just sorts each categorical variable by the frequency of each value. For the value with the largest frequency, encode it to $0$, then $1,2,3...$ as the frequency decreases.

### joint_encode

We pass a dataframe and a list which contains the variable indices of the categorical variables into the function. Then, we sort the value combinations by their frequencies in descending order. Later on, we iterate over the value combinations and encode the variables from $0$ onwards. For a value combination, if one feature value has already been encoded, then discard this combination. If all values for a feature has been encoded, then disregard it and just look at the other ones. More details can be found in the script.

### outlier_removal

We only consider the continuous variables here. The distance is calculated by Mahalanobis distance. 



## Demo

```python
df = pd.read_csv("xxx.csv")
cont_col = [1,2,3,4]
unrankable = [5,6,7,8]
discrete_feature_val = {k:df.iloc[:,k].nunique() for k in unrankable}

df = utils.joint_encode(df, unrankable)
# for encoding and clustering of categorical variables
cate_val = df.iloc[:,unrankable].to_numpy()
corr_matrix = stats.spearmanr(cate_val)[0]
abs_corr = np.absolute(corr_matrix)
distance_matrix = 1 - abs_corr
clusterer = AgglomerativeClustering(affinity='precomputed', linkage='average', 
                                    distance_threshold=0.3, n_clusters=None)
clusterer.fit(distance_matrix)
cluster_dict = dict()
for i,c in enumerate(clusterer.labels_):
    if c not in cluster_dict.keys():
        cluster_dict[c] = list()
    cluster_dict[c].append(cate_col[i])
cate_cluster_book = list()
for k in cluster_dict.keys():
    cate_cluster_book.append(cluster_dict[k].copy())
cate_clusters = list()
for c in cate_cluster_book:
    if len(c) > 1:
        cate_clusters.append(c)
# scaling & outlier removal        
scaler = preprocessing.MinMaxScaler()
cols_to_norm = ["X"+str(i) for i in cont_col]
df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])
reduced_df = utils.outlier_removal(df, cont_col)

X = reduced_df[colnames[:-1]].to_numpy()
Y = reduced_df[colnames[-1]].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state = 40)

ciber = cm.clustered_comonotonic(x_train, y_train,
                                 discrete_feature_val,cont_col,categorical, 
                                 0.92, cate_clusters, corrtype='spearman',
                                 discrete_method='mdlp')
ciber.run()
ciber_predict = ciber.predict(x_test)
```

