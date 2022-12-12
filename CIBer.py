import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import warnings
from CIBer_Engineering import Discretization, Joint_Encoding, Frequency_Encoding
import concurrent.futures
import pickle
copy_obj = lambda obj: pickle.loads(pickle.dumps(obj))

class CIBer():
    def __init__(self, cont_col=[], asso_method='kendall', min_asso=0.8, alpha=1, 
                 disc_method="mdlp", joint_encode=True, **kwargs):
        self.cont_col = cont_col
        if asso_method.lower() == "total_order":
            self.asso_method = self._total_order
        else:
            self.asso_method = asso_method.lower()
        self.min_asso = min_asso
        self.alpha = alpha
        self.disc_method = disc_method.lower()
        self.discretizer = Discretization(cont_col, self.disc_method, **kwargs)
        if joint_encode:
            self.encoder = Joint_Encoding()
        else:
            self.encoder = Frequency_Encoding()

        self.cluster_book = dict()
        assert asso_method.lower() in ["spearman", "pearson", "kendall", "total_order"]
        assert min_asso >= 0 and min_asso <= 1
        assert alpha > 0
    
    def _total_order(self, a, b):
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
        
    def _get_association(self, x_train):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            asso_matrix = pd.DataFrame(data=x_train).corr(method=self.asso_method).to_numpy()
        
        distance_matrix = np.nan_to_num(1 - np.absolute(asso_matrix), nan=1)
        AGNES = AgglomerativeClustering(affinity='precomputed', linkage='complete', 
                                        distance_threshold=1-self.min_asso, n_clusters=None)
        AGNES.fit(distance_matrix)
        AGNES_clusters = AGNES.labels_
        return sorted([np.where(AGNES_clusters == cluster)[0].tolist() for cluster in np.unique(AGNES_clusters)])
    
    def _get_prior_prob(self, y_train):
        # prior_prob: dict() key is class, value is the prior probability of this class
        # class_idx:  dict() key is class, value is a list containing the indices of instances for this class
        classes, counts = np.unique(y_train, return_counts=True)
        self.prior_prob = dict(zip(classes, counts/len(y_train)))
        self.class_idx = {k: np.where(y_train == k)[0].tolist() for k in classes}
        self.y_cate = pd.Categorical(y_train, categories=np.unique(y_train))
    
    def fit(self, x_train, y_train):
        ncol = np.shape(x_train)[1]
        self.cate_col = list(set(np.arange(ncol)) - set(self.cont_col))
        self._get_prior_prob(y_train)
        
        if len(self.cont_col) > 0:
            x_train = self.discretizer.fit_transform(x_train, y_train)
            
        if len(self.cate_col) > 0:
            x_train[:,self.cate_col] = self.encoder.fit(x_train[:,self.cate_col])
        
        self.transform_x_train = x_train
        
        for c, idx in self.class_idx.items():
            self.cluster_book[c] = self._get_association(self.transform_x_train[idx,:])
    
    def _get_prob_dist_single(self, x):
        y_val = []
        for c in self.class_idx.keys():
            prob = self.prior_prob[c]
            for cluster in self.cluster_book[c]:
                if len(cluster) == 1:
                    x_col = cluster[0]
                    if cluster[0] in self.cont_col and self.disc_method == "ndd":
                        total_bins = len(self.cond_prob[x_col][c])
                        lower_idx = int(np.clip(x[x_col] - 1, 0, total_bins - 1))
                        upper_idx = int(np.clip(x[x_col] + 1, 1, total_bins))
                        prob *= sum(list(self.cond_prob[x_col][c].values())[lower_idx:(upper_idx+1)])
                    else:
                        prob *= self.cond_prob[x_col][c][x[x_col]]
                else:
                    sup_cluster, inf_cluster = [], []
                    for col in cluster:
                        x_index = list(self.cond_prob[col][c]).index(x[col])
                        item_prob = [0] + list(self.cond_cum_prob[col][c].values())
                        total_bins = len(item_prob) - 1
                        if col in self.cont_col and self.disc_method == "ndd":
                            lower_idx = int(np.clip(x_index - 1, 0, total_bins - 1))
                            upper_idx = int(np.clip(x_index + 1, 1, total_bins))
                            inf_cluster.append(item_prob[upper_idx])
                            sup_cluster.append(item_prob[lower_idx])
                        else:
                            inf_cluster.append(item_prob[x_index+1])
                            sup_cluster.append(item_prob[x_index])
                    
                    inf = min(inf_cluster)
                    sup = max(sup_cluster)
                    prob *= max(inf - sup, 1e-5)
            y_val.append(prob)

        return y_val
    
    def _get_cond_prob(self, x_train, x_test):
        ncol = np.shape(x_train)[1]
        self.cond_prob = dict() # key is column, value dict: key is class, value is corresponding probabilities
        self.cond_cum_prob = dict() # key is column, value dict: key is class, value is corresponding cumulative probabilities
        for col in range(ncol):
            categories = np.unique(np.append(x_train[:,col], x_test[:,col]))
            x_cate = pd.Categorical(x_train[:,col], categories=categories)
            Laplace_tab = pd.crosstab(x_cate, self.y_cate, dropna=False) + self.alpha
            density_tab = Laplace_tab.apply(lambda x: x/x.sum())
            self.cond_prob[col] = density_tab.to_dict()
            self.cond_cum_prob[col] = density_tab.cumsum().to_dict()
        
    def predict_proba(self, x_test):
        if len(self.cont_col) > 0:
            x_test = self.discretizer.transform(x_test)
        
        if len(self.cate_col) > 0:
            x_test[:,self.cate_col] = self.encoder.transform(x_test[:, self.cate_col])
        
        self.transform_x_test = x_test
        self._get_cond_prob(self.transform_x_train, self.transform_x_test)

        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            y_val = [executor.submit(self._get_prob_dist_single, x).result() for x in x_test]
        return np.array(y_val)/np.sum(y_val, axis=1).reshape(-1, 1)
    
    def predict(self, x_test):
        y_proba = self.predict_proba(x_test)
        class_label = np.array(list(self.class_idx.keys()))
        return class_label[list(np.argmax(y_proba, axis=1))]