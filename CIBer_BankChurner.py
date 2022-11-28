# Example Code for CIBer

import os
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from CIBer import CIBer

import pickle
copy = lambda obj: pickle.loads(pickle.dumps(obj))

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
simplefilter("ignore", category=UndefinedMetricWarning)
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=RuntimeWarning)

#%%
# Function for model assessment

def _get_model_performance(name, y_true, y_pred, y_prob):
    return [name, roc_auc_score(y_true, y_prob[:,1]), 
            precision_score(y_true, y_pred), 
            recall_score(y_true, y_pred), 
            f1_score(y_true, y_pred), 
            accuracy_score(y_true, y_pred)]

#%%
dataset = "BankChurners"
df = pd.read_csv(f"Dataset/{dataset}.csv")

# Drop unused columns
df.drop(columns=['CLIENTNUM', 
                 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], inplace=True)

# Encode categorical feature values into discrete ones
df['Attrition_Flag'] = pd.factorize(df['Attrition_Flag'])[0] + 0
df['Gender'] = pd.factorize(df['Gender'])[0] + 1
df['Education_Level'] = pd.factorize(df['Education_Level'])[0] + 1
df['Marital_Status'] = pd.factorize(df['Marital_Status'])[0] + 1
df['Income_Category'] = pd.factorize(df['Income_Category'])[0] + 1
df['Card_Category'] = pd.factorize(df['Card_Category'])[0] + 1

# Stack label y as the last column of df
cols = df.columns.to_list()
cols.append(cols[0])
cols = cols[1:]
df = df[cols]

#%%
# parameters to be added in CIBer
cont_col = [7, 11, 12, 13, 14, 15, 16, 17, 18]
categorical = list(set(np.arange(len(df.columns)-1)) - set(cont_col))
min_asso = 0.95

# Dataset preparation
label_name = "Attrition_Flag"
n_sample = 7000
n_test = 200

np.random.seed(4012)
idx_train = np.random.choice(np.arange(len(df)), n_sample, replace=False)
X_train = df.iloc[idx_train,:-1].to_numpy()
y_train = df.iloc[idx_train,-1].to_numpy()

# obtain test. 1500 in EACH class
samples_per_group_dict = {0:n_test, 1:n_test}
df_test = df[~df.index.isin(idx_train)]
df_test = df_test.groupby(label_name).apply(lambda group: group.sample(samples_per_group_dict[group.name])).reset_index(drop=True)
X_test = df_test.iloc[:,:-1].to_numpy()
y_test = df_test.iloc[:,-1].to_numpy()

#%%
# Fit CIBer
CIBer_clf = CIBer(cont_col=cont_col, asso_method='total_order', min_asso=min_asso, 
                  joint_encode=False, disc_method="norm", n_bins=50)
CIBer_clf.fit(X_train, y_train)
CIBer_predict = CIBer_clf.predict(X_test)
CIBer_proba = CIBer_clf.predict_proba(X_test)


print(_get_model_performance('CIBer (auto)', y_test, CIBer_predict, CIBer_proba))