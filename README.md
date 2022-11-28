# Repository for CIBer

Paper: A New Simple Effective InsurTech Tool: Comonotone-Independence Bayes classifier (CIBer)
Author: Yongzhao CHEN, Ka Chun CHEUNG, Nok Sang FAN, James Cheng PENG, and Sheung Chi Phillip YAM

# User Guide for CIBer

This is the user guide for Comonotone-Independence Bayesian Classifier (CIBer). CIBer is a supervised learning model which deals with multi-class classification tasks. The document consists of two parts: the requirements for the input data and the input parameters in CIBer.py.


## Data Requirements

CIBer deals with multi-class classification tasks with numerical or discrete (but should be ordered) input variables. Before passing the data into the model, please make sure to do some proper preprocessing beforehand, e.g. removals of outlier and missing observation. If your dataset contains categorical data like gender, nationality, occupation, which are not ordered, please refer to the function $\textbf{joint_encode}$ in the CIBer_Engineering.py. 


## CIBer.py

### CIBer

$\texttt{cont_col}$: a list, containing the indices of the continuous variables

$\texttt{asso_method}$: a string can be set to "pearson", "spearman", "kendall", "total_order". Four measurements to correlation. The default is "total_order"

$\texttt{min_asso}$: a number between $0,1$ which specifies the threshold of correlation when determining the comonotonic relationship. The default value is 0.8

$\texttt{alpha}$: a positive number used in Laplacian smoothing. The default value is 1

$\texttt{group_cate}$: a boolean, whether to combine multiple categories with same label into one group. The default value is False

$\texttt{joint_encode}$: a boolean, whether to use joint encoding. The default value is True

$\texttt{disc_method}$: a string indicating the discretization method adopted for each continuous feature variable. The default string is "auto"

$\texttt{n_bins}: a positive integer for the total number of bins for each discretization. The default value is 10

$\texttt{disc_backup}$: a string indicating the discretization method adopted if the method $\texttt{disc_method="mdlp"}$ fails. The default string is "pkid"
