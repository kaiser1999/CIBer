import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import warnings
from CIBer_Engineering import Discretization, Joint_Encoding, Frequency_Encoding, Group_Categorical

class CIBer():
    def __init__(self, cont_col=[], asso_method='kendall', min_asso=0.8, alpha=1, 
                 group_cate=False, joint_encode=True, disc_method="mdlp", n_bins=10, disc_backup="pkid"):
        self.min_asso = min_asso
        self.group_cate = group_cate
        self.disc_method = disc_method.lower()
        self.discretizer = Discretization(cont_col, disc_method, disc_backup, n_bins)
        
        self.grouping = Group_Categorical(cont_col)
        self.cont_col = cont_col
        if joint_encode:
            self.encode = Joint_Encoding()
        else:
            self.encode = Frequency_Encoding()
        self.cluster_book = dict()
        
        assert asso_method.lower() in ["spearman", "pearson", "kendall", "total_order"]
        if asso_method.lower() == "total_order":
            self.asso_method = self.total_order
        else:
            self.asso_method = asso_method.lower()
        
        assert min_asso >= 0 and min_asso <= 1
        self.alpha = alpha
        assert self.alpha > 0
    
    def total_order(self, a, b):
        freq_df = pd.DataFrame({"a":a, "b":b}).groupby(['a','b']).size().reset_index(name='count')
        x, y, count = freq_df.T.values
        n = len(a)
        
        # If most of the numbers are of the same group, they should not be considered as a cm cluster
        if len(np.unique(x)) <= 2 or len(np.unique(y)) <= 2:  # Not accept Binary as cm cluster
            return 0
        
        if len(x) < 10000:      # Avoid Memory Error
            mat = np.sign((x[:, None] - x)*(y[:, None] - y))
            mat[mat == 0] = 1
            mat *= (count[:, None] * count)
            np.fill_diagonal(mat, count*(count-1)/2)
            return np.sum(np.tril(mat, k=0))/(n*(n-1)/2)
        else:
            s = np.sum(count*(count - 1)/2)
            for i in range(1, len(x)):
                arr = np.sign((x[i] - x[:i]) * (y[i] - y[:i]))
                arr[arr == 0] = 1
                arr *= (count[i] * count[:i])
                s += np.sum(arr)
            return s / (n*(n-1)/2)
        
    def association(self, x_train, c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            asso_matrix = pd.DataFrame(data=x_train).corr(method=self.asso_method).to_numpy()
        
        distance_matrix = 1 - np.absolute(asso_matrix)
        distance_matrix = np.nan_to_num(distance_matrix, nan=1)
        
        AGNES = AgglomerativeClustering(affinity='precomputed', linkage='complete', 
                                        distance_threshold=1-self.min_asso, n_clusters=None)
        AGNES.fit(distance_matrix)
        self.cluster_book[c] = list()
        for cluster in np.unique(AGNES.labels_):
            self.cluster_book[c].append(np.where(AGNES.labels_ == cluster)[0].tolist())
        
        self.cluster_book[c] = sorted(self.cluster_book[c])
    
    def get_prior_prob(self, y_train):
        # get the prior probability and indices of instances for different classes
        prior_prob = dict() # key is class, value is the prior probability of this class
        class_idx = dict() # key is class, value is a list containing the indices of instances for this class
        for value in np.unique(y_train):
            class_idx[value] = np.squeeze(np.where(y_train == value)).tolist()
        prior_prob = {k:len(v)/len(y_train) for k,v in class_idx.items()}
        self.prior_prob = prior_prob
        self.class_idx = class_idx
    
    def Laplacian_Correction(self, x_train, x_test, col, class_dict):
        new_key = False
        # Laplacian correction for training and testing dataset
        for k in np.unique(list(x_train[:,col]) + list(x_test[:,col])):
            if k not in class_dict.keys():
                new_key = True
                class_dict[k] = 0
            
        if new_key:
            for key in class_dict.keys():
                class_dict[key] += self.alpha
        
        # M-estimate
        for key in class_dict.keys():
            p = class_dict[key]/sum(class_dict.values())
            class_dict[key] += self.alpha * p
        return class_dict
    
    def get_cond_prob(self, x_train, x_test):
        ncol = np.shape(x_train)[1]
        
        self.cond_prob = dict() # key is column, value dict: key is class, value is corresponding probabilities
        self.cond_cum_prob = dict() # key is column, value dict: key is class, value is corresponding cumulative probabilities
        for col in range(ncol):
            feature_dict = dict()
            cum_feature_dict = dict()
            for c in self.class_idx.keys():
                indices = self.class_idx[c]
                values, counts = np.unique(x_train[indices, col], return_counts=True)
                
                class_dict = dict(zip(values, counts))
                class_dict = self.Laplacian_Correction(x_train, x_test, col, class_dict)
                    
                summation = sum(class_dict.values())
                cum_sum = np.cumsum(list(class_dict.values()))
                
                feature_dict[c] = {k:v/summation for k,v in class_dict.items()}
                cum_feature_dict[c] = dict(zip(class_dict.keys(), cum_sum/summation))
            self.cond_prob[col] = feature_dict
            self.cond_cum_prob[col] = cum_feature_dict
    
    def fit(self, x_train, y_train):
        x_train = self.discretizer.fit_transform(x_train, y_train)
        if self.group_cate:
            x_train = self.grouping.fit(x_train, y_train)
            
        ncol = np.shape(x_train)[1]
        self.cate_col = list(set(np.arange(ncol)) - set(self.cont_col))
        if len(self.cate_col) > 0:
            x_train[:,self.cate_col] = self.encode.fit(x_train[:,self.cate_col])
        
        self.transform_x_train = x_train
        
        for value in np.unique(y_train):
            self.association(self.transform_x_train[y_train == value, :], value)
        self.get_prior_prob(y_train)
        
    def get_prob_dist_single(self, x):
        y_prob = []
        for c in self.class_idx.keys():
            prob = self.prior_prob[c]
            for cluster in self.cluster_book[c]:
                if len(cluster) == 1:
                    if cluster[0] in self.cont_col and self.disc_method == "ndd":
                        total_bins = len(self.cond_prob[cluster[0]][c])
                        lower_idx = int(np.clip(x[cluster[0]] - 1, 0, total_bins - 2))
                        upper_idx = int(np.clip(x[cluster[0]] + 1, 2, total_bins - 1))
                        prob *= sum(list(self.cond_prob[cluster[0]][c].values())[lower_idx:(upper_idx+1)])
                    else:
                        prob *= self.cond_prob[cluster[0]][c][x[cluster[0]]]
                else:
                    sup_cluster, inf_cluster = [], []
                    for col in cluster:
                        sup_index = list(self.cond_prob[col][c]).index(x[col])
                        item_prob = [0] + list(self.cond_cum_prob[col][c].values())
                        
                        if col in self.cont_col and self.disc_method == "ndd":
                            lower_idx = int(np.clip(sup_index-1, 0, len(item_prob) - 2))
                            upper_idx = int(np.clip(sup_index+1, 2, len(item_prob)))
                            inf_cluster.append(item_prob[upper_idx])
                            sup_cluster.append(item_prob[lower_idx])
                        else:
                            inf_cluster.append(item_prob[sup_index+1])
                            sup_cluster.append(item_prob[sup_index])
                    
                    inf = min(inf_cluster)
                    sup = max(sup_cluster)
                    prob *= max(inf - sup, 1e-5)
            y_prob.append(prob)
        
        y_prob = np.array(y_prob)/np.sum(y_prob)
        return y_prob
    
    def get_proba(self, x_test):
        self.get_cond_prob(self.transform_x_train, x_test)
        y_predict = list()
        for x in x_test:
            y_predict.append(self.get_prob_dist_single(x))
        return np.array(y_predict)
    
    def get_transform(self, x_test):
        x_test = self.discretizer.transform(x_test)
        if self.group_cate:
            x_test = self.grouping.transform(x_test)
        if len(self.cate_col) > 0:
            x_test[:,self.cate_col] = self.encode.transform(x_test[:,self.cate_col])
        return x_test
    
    def predict(self, x_test):
        self.transform_x_test = self.get_transform(x_test)
        class_label = np.array(list(self.class_idx.keys()))
        return class_label[list(np.argmax(self.get_proba(self.transform_x_test), axis=1))]
    
    def predict_proba(self, x_test):
        self.transform_x_test = self.get_transform(x_test)
        return self.get_proba(self.transform_x_test)