# Repository for CIBer

Paper: A New Simple Effective InsurTech Tool: Comonotone-Independence Bayes classifier (CIBer)
Author: Yongzhao CHEN, Ka Chun CHEUNG, Nok Sang FAN, James Cheng PENG, and Sheung Chi Phillip YAM

This is the user guide for Comonotone-Independence Bayesian Classifier (CIBer). CIBer is a supervised learning model which deals with multi-class classification tasks. The continuous feature variables are discretized and those categorical ones are encoded via the proposed **Joint Encoding**. 

This document mainly explain the important and practical functions in **CIBer.py** and **CIBer_Engineering.py**. Lastly, the **CIBer_Bankchurner.py** gives a simple but illuminating example on CIBer with the use of [Bankchurner](https://www.kaggle.com/code/thomaskonstantin/bank-churn-data-exploration-and-churn-prediction/data) dataset by Thomas Konstantin.

# Remarks
The **MDLP** discretization method has been disabled, since it requires additional package.
## window users
1. install visual studio community, and then install [**Microsoft C++ Build Tools**](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for C/C++ related packages
2. type the following line in terminal to install
> pip install mdlp

## macos users
1. type the following line in terminal to install the **Command Line Tools package**
> xcode-select --install
2. type the following line in terminal to install
> pip install mdlp

# Data Requirements

CIBer deals with multi-class classification tasks with numerical or discrete (but should be ordered) input variables. Before passing the data into the model, please perform some proper data preprocessing beforehand, e.g. removals of outlier and missing observation, and encode all categorical feature variables with numerical values.

# CIBer.py

## CIBer

> init(self, cont_col=[], asso_method='kendall', min_asso=0.8, alpha=1, disc_method="mdlp", joint_encode=True, **kwargs)

**_cont_col_**: a list, containing the indices of the continuous variables

**_asso_method_**: a string can be set to "pearson", "spearman", "kendall", "total_order". Four measurements to correlation. The default is "kendall"

**_min_asso_**: a number between $0,1$ which specifies the threshold of correlation when determining the comonotonic relationship. The default value is 0.8

**_alpha_**: a positive number used in Laplacian smoothing. The default value is 1

**_joint_encode_**: a boolean, whether to use joint encoding. The default value is True

**_disc_method_**: a string indicating the discretization method adopted for each continuous feature variable. The default string is "auto"

**_**kwargs_**: additional keyworded arguments passing to **Discretization()**, below are two acceptable keyworded arguments

**_n_bins_**: a positive integer for the total number of bins for each discretization.

**_disc_backup_**: a string indicating the discretization method adopted if the method **_disc_method="mdlp"_** fails.

> fit(self, x_train, y_train)

**_x_train_**: a numpy $n \times p$ array for the $p$ training (real-valued) feature variables with $n$ training observations

**_y_train_**: a numpy $n \times 1$ array for the $n$ training (real-valued) labels

> predict(self, x_test)

**_x_test_**: a numpy $n \times p$ array for the $p$ test (real-valued) feature variables with $n$ test observations

**return**: a numpy $n \times 1$ array for the $n$ predicted class labels

> predict_proba(self, x_test)

**_x_test_**: a numpy $n \times p$ array for the $p$ test (real-valued) feature variables with $n$ test observations

**return**: a numpy $n \times K$ array for the predicted probabilities of the $K$ classes with $n$ test observations

# CIBer_Engineering.py

> Discretization(cont_col, disc_method, disc_backup="pkid", n_bins=10)

**_cont_col_**: a list of indices to be discretized

**_disc_method_**: any string in **DISC_BASE** + **SCIPY_DIST**, (refer to CIBer.py)

list of distributions provided by scipy used in Equal-quantile distribution method, number of bins determined by **_n_bins_**
> SCIPY_DIST = ["uniform", "norm", "t", "chi2", "expon", "laplace", "skewnorm", "gamma"]

list of common discretiztion methods for Na\"ive Bayes classifier
> SIZE_BASE = ["equal_size", "pkid", "ndd", "wpkid"]

list of all discretization methods except **SCIPY_DIST**
> DISC_BASE = ["equal_length", "auto"] + SIZE_BASE

list of alternative discretization methods if mdlp fails except **SCIPY_DIST**
> MDLP_BACKUP = ["equal_length", "auto"] + SIZE_BASE

**return** a class for discretization method

## Joint_Encoding

> init(self, df, col_index)

**_df_**: a $n \times p$ dataframe for $p$ feature variables of $n$ observations

**_col_index_**: a list, containing the indices of categorical feature variables

> fit(self, x_train)

**_x_train_**: a $n \times p$ numpy array for the $p$ training feature variables with $n$ training observations

> transform(self, x_test)

**_x_test_**: a numpy $n \times p$ array for the $p$ test (real-valued) feature variables with $n$ test observations

**return**: a numpy $n \times p$ array with the encoded categorical feature variable
