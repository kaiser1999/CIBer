# Repository for CIBer

Paper: A New Simple Effective InsurTech Tool: Comonotone-Independence Bayes classifier (CIBer)
Author: Yongzhao CHEN, Ka Chun CHEUNG, Nok Sang FAN, James Cheng PENG, and Sheung Chi Phillip YAM

# User Guide for CIBer

This is the user guide for Comonotone-Independence Bayesian Classifier (CIBer). CIBer is a supervised learning model which deals with multi-class classification tasks. The document consists of two parts: the requirements for the input data and the input parameters in CIBer.py.


## Data Requirements

CIBer deals with multi-class classification tasks with numerical or discrete (but should be ordered) input variables. Before passing the data into the model, please make sure to do some proper preprocessing beforehand, e.g. removals of outlier and missing observation. If your dataset contains categorical data like gender, nationality, occupation, which are not ordered, please refer to the class **_Joint_Encoding_** in the CIBer_Engineering.py. 


## CIBer.py

### CIBer

#### init

**_cont_col_**: a list, containing the indices of the continuous variables

**_asso_method_**: a string can be set to "pearson", "spearman", "kendall", "total_order". Four measurements to correlation. The default is "total_order"

**_min_asso_**: a number between $0,1$ which specifies the threshold of correlation when determining the comonotonic relationship. The default value is 0.8

**_alpha_**: a positive number used in Laplacian smoothing. The default value is 1

**_group_cate_**: a boolean, whether to combine multiple categories with same label into one group. The default value is False

**_joint_encode_**: a boolean, whether to use joint encoding. The default value is True

**_disc_method_**: a string indicating the discretization method adopted for each continuous feature variable. The default string is "auto"

**_n_bins_**: a positive integer for the total number of bins for each discretization. The default value is 10

**_disc_backup_**: a string indicating the discretization method adopted if the method **_disc_method="mdlp"_** fails. The default string is "pkid"

### fit

**_x_train_**: a numpy $n \times p$ array for the $p$ training (real-valued) feature variables with $n$ training observations

**_y_train_**: a numpy $n \times 1$ array for the $n$ training (real-valued) labels

### predict

**_x_test_**: a numpy $n \times p$ array for the $p$ test (real-valued) feature variables with $n$ test observations

**return**: a numpy $n \times 1$ array for the $n$ predicted class labels

### predict_proba

**_x_test_**: a numpy $n \times p$ array for the $p$ test (real-valued) feature variables with $n$ test observations

**return**: a numpy $n \times K$ array for the predicted probabilities of the $K$ classes with $n$ test observations
