import os, sys, scipy
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed
import warnings
from .CIBer_Engineering import Discretization, Joint_Encoding
import dill
copy_obj = lambda obj: dill.loads(dill.dumps(obj))

sys.path.append(os.path.dirname(scipy.stats.__file__))                          # read _stats.pyd
from _stats import _kendall_dis

class CIBer():
    def __init__(self, cont_col, asso_method='modified', min_asso=0.95, alpha=1, 
                 disc_method="norm", n_jobs=None, **kwargs):
        self.cont_col = cont_col
        if callable(asso_method):
            self.asso_method = asso_method
        elif asso_method.lower() == "modified":
            self.asso_method = self._modified_tau
        else:
            self.asso_method = asso_method
        self.min_asso = min_asso
        self.alpha = alpha
        self.disc_method = disc_method
        self.discretizer = Discretization(cont_col, self.disc_method, **kwargs)
        self.encoder = Joint_Encoding()
        self.n_jobs = None if n_jobs is None else int(n_jobs)
        
        self.distance_matrix_ = dict()
        self.cluster_book = dict()
        if isinstance(asso_method, str): assert asso_method in ["spearman", "pearson", "kendall", "modified"]
        assert min_asso >= 0 and min_asso <= 1
        assert alpha > 0
        
        # Can be passed to sklearn.ensemble AdaBoostClassifier(estimator=CIBer(...))
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._param_names = ['cont_col', 'asso_method', 'min_asso', 'alpha', 'disc_method'] + list(kwargs.keys())
    
    @staticmethod
    def _modified_tau(x, y):    # modified from scipy.stats._stats_py.kendalltau
        # If most of the numbers are of the same group, they should not be considered as a cm cluster
        if len(np.unique(x)) <= 2 or len(np.unique(y)) <= 2:  # Not accept Binary as cm cluster
            return 0

        # sort on y and convert y to dense ranks
        perm = np.argsort(y)
        x, y = x[perm], y[perm]
        y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

        # stable sort on x and convert x to dense ranks
        perm = np.argsort(x, kind='mergesort')
        x, y = x[perm], y[perm]
        x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)
        
        dis = _kendall_dis(x, y)                # number of discordant pairs
        tot = (x.size * (x.size - 1)) // 2
        return min(1., max(-1., (tot - 2*dis)/tot))
    
    def _get_corr(self, X_train, method):
        _, n_col = X_train.shape
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(method)(X_train[:, i], X_train[:, j])
            for i in range(n_col-1) for j in range(i+1, n_col)
        )
        
        corr_matrix, k = np.diag(np.ones(n_col)), 0
        for i in range(n_col-1):
            for j in range(i+1, n_col):
                corr_matrix[i, j] = results[k]
                corr_matrix[j, i] = results[k]
                k += 1

        return corr_matrix
        
    def _get_association(self, X_train):
        if self.n_jobs is not None and callable(self.asso_method):
            asso_matrix = self._get_corr(X_train, method=self.asso_method)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                asso_matrix = pd.DataFrame(data=X_train).corr(method=self.asso_method).to_numpy()
        
        distance_matrix = np.nan_to_num(1 - np.absolute(asso_matrix), nan=1)
        AGNES = AgglomerativeClustering(metric='precomputed', linkage='complete', 
                                        distance_threshold=1-self.min_asso, n_clusters=None)
        AGNES.fit(distance_matrix)
        AGNES_clusters = AGNES.labels_
        return distance_matrix, sorted([np.where(AGNES_clusters == cluster)[0].tolist() for cluster in np.unique(AGNES_clusters)])
    
    def _get_prior_prob(self, y_train):
        # prior_prob: dict() key is class, value is the prior probability of this class
        classes, inverse = np.unique(y_train, return_inverse=True)      # return_inverse so that y_train can now take negative integers
        counts = np.bincount(inverse, weights=self.sample_weight)
        self.prior_prob = dict(zip(classes, counts/sum(counts)))
        self.y_cate = pd.Categorical(y_train, categories=classes)
    
    def fit(self, X_train, y_train, sample_weight=None):
        nrow, ncol = np.shape(X_train)
        self.sample_weight = np.ones(nrow) if sample_weight is None else nrow*sample_weight/sum(sample_weight)  # same weight on alpha
        assert len(self.sample_weight) == nrow
        
        self.cate_col = list(set(np.arange(ncol)) - set(self.cont_col))
        self.encoder.cate_col = self.cate_col
        
        if len(self.cont_col) > 0:
            X_train = self.discretizer.fit_transform(X_train, y_train)
        
        if len(self.cate_col) > 0:
            X_train = self.encoder.fit_transform(X_train)
        
        self.transform_X_train = X_train
        self.y_train = y_train
        # class_idx:  dict() key is class, value is a list containing the indices of instances for this class
        self.class_idx = {k: np.where(y_train == k)[0].tolist() for k in np.unique(y_train)}
        for c, idx in self.class_idx.items():
            self.distance_matrix_[c], self.cluster_book[c] = self._get_association(self.transform_X_train[idx,:])
        
        self.classes_ = np.array(list(self.class_idx.keys()))
        self.n_classes = len(self.classes_)
    
    def _get_cond_prob(self, X_train, X_test):
        ncol = np.shape(X_train)[1]
        self.cond_prob = dict()     # key is column, value dict: key is class, value is corresponding probabilities
        self.cond_cum_prob = dict() # key is column, value dict: key is class, value is corresponding cumulative probabilities
        self.prev_idx = dict()      # key is column, value dict: key is value, value is previous value
            
        for col in range(ncol):
            categories = np.unique(np.append(X_train[:,col], X_test[:,col]))
            x_cate = pd.Categorical(X_train[:,col], categories=categories)
            Laplace_tab = pd.crosstab(x_cate, self.y_cate, self.sample_weight, aggfunc="sum", dropna=False) + self.alpha
            density_tab = Laplace_tab.apply(lambda x: x/x.sum())
            
            if col in self.cont_col and self.disc_method == "ndd":
                density_tab = density_tab.rolling(window=3, min_periods=2, center=True).sum()
                density_tab = density_tab / density_tab.sum(axis=0)
                
            density_tab.loc[-1.0] = 0
            idx_lst = sorted(density_tab.index)
            density_tab = density_tab.reindex(index=idx_lst)
            self.cond_prob[col] = density_tab.to_dict()
            self.cond_cum_prob[col] = density_tab.cumsum().to_dict()
            self.prev_idx[col] = dict(zip(idx_lst[1:], idx_lst[:-1]))
        
    def predict_proba(self, X_test):
        if len(self.cont_col) > 0:
            X_test = self.discretizer.transform(X_test)
        
        if len(self.cate_col) > 0:
            X_test = self.encoder.transform(X_test)
        
        self.transform_X_test = X_test
        self._get_prior_prob(self.y_train)
        self._get_cond_prob(self.transform_X_train, self.transform_X_test)
        
        y_val = []
        for c in self.cluster_book.keys():
            indep_prob = {cluster[0]: self.cond_prob[cluster[0]][c] for cluster in self.cluster_book[c] if len(cluster) == 1}
            clust_prob = [{col: self.cond_cum_prob[col][c] for col in cluster} for cluster in self.cluster_book[c] if len(cluster) > 1]

            df_test = pd.DataFrame(X_test)
            prob = self.prior_prob[c] * df_test[indep_prob.keys()].replace(indep_prob).prod(axis=1)

            for comon_prob in clust_prob:
                prob_inf = df_test[comon_prob.keys()].replace(comon_prob).min(axis=1)
                prob_sup = df_test[comon_prob.keys()].replace(self.prev_idx).replace(comon_prob).max(axis=1)
                prob = prob * np.maximum(prob_inf - prob_sup, 1e-5)
            
            y_val.append(prob)
        
        # y_val is cond. likelihood, consider F_k is cond. log-llh F_{nk} = log(y_val_{nk}), then p_{nk} = y_val_{nk} / sum_k (y_val_{nk})
        return np.array(y_val).T / np.sum(y_val, axis=0).reshape(-1, 1)
    
    def predict(self, X_test):
        y_proba = self.predict_proba(X_test)
        return self.classes_[list(np.argmax(y_proba, axis=1))]
    
    def get_params(self, deep=True):
        return {param: getattr(self, param) for param in self._param_names}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

#%%
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score as CHIndex
from tqdm import tqdm, trange
from datetime import datetime as dt
             
def MAE(y_test, y_pred):
    return np.mean(np.abs(y_test - y_pred))

def MSE(y_test, y_pred):
    return np.mean((y_test - y_pred)**2)          

class KMeansRegressor():
    def __init__(self, clf, K_max=50, n_trials=5, seed=4012, K_min=None, loss="mae", verbose=True):
        self.clf = clf
        self.K_max, self.n_trials, self.seed = int(K_max), int(n_trials), int(seed)
        self.K_min = 2 if K_min is None else int(K_min)
        self.loss = loss
        self.verbose = verbose
        
        assert loss.lower() in ["mae", "mse"]
        self.loss_func = {'mae': MAE, 'mse': MSE}[loss.lower()]

    # Try KMeans() several times and output the best trial
    def _get_best_km(self, y_train, K):
        if self.seed is not None: np.random.seed(self.seed)
        best_r, best_km = 0, None
        for i in range(self.n_trials):
            km = KMeans(n_clusters=K, n_init="auto").fit(y_train)
            r = CHIndex(y_train, km.labels_)
            if r > best_r:                                                  # update best_r if it is less than r
                best_r, best_km = r, km
                
        return best_km
    
    @staticmethod
    def _get_predict(X_test, clf, km):
        centers = km.cluster_centers_[np.unique(km.labels_)]                # remove centers that have no count; np.unique already sorted
        return (clf.predict_proba(X_test) @ centers).flatten()
        
    def fit(self, X_train, y_train, sample_weight=None):
        best_loss, best_clf, best_km, best_K = np.inf, None, None, None
        for K in (pbar := trange(self.K_min, self.K_max+1)) if self.verbose else range(self.K_min, self.K_max+1):
            km = self._get_best_km(y_train.reshape(-1, 1), K)
            clf = copy_obj(self.clf)
            clf.fit(X_train, km.labels_, sample_weight)
            y_pred = self._get_predict(X_train, clf, km)
            loss = self.loss_func(y_train, y_pred)

            if loss < best_loss:                                            # update best_loss if it is larger than loss
                best_loss, best_clf, best_km, best_K = loss, clf, km, K
            
            if self.verbose:
                pbar.set_postfix_str(f"K={K}, {self.loss.upper()}={loss:.4}; " + 
                                     f"best K={best_K}, best {self.loss.upper()}={best_loss:.4}")
        
        self.best_clf, self.best_km = best_clf, best_km
    
    def predict(self, X_test):
        return self._get_predict(X_test, self.best_clf, self.best_km)
        

#%%
class AdaBoostCIBer(CIBer):
    def __init__(self, n_CIBer=5, verbose=True, **kwargs):
        self.n_CIBer = int(n_CIBer)
        self.verb = lambda lst: tqdm(lst) if verbose else lst
        
        self.base_CIBer = CIBer(**kwargs)
        self.beta = np.ones(self.n_CIBer)
        
    def fit(self, X_train, y_train, sample_weight=None):
        self.base_CIBer.fit(X_train, y_train, sample_weight)
        self.classes_ = self.base_CIBer.classes_
        self.CIBer_lst = [copy_obj(self.base_CIBer) for i in range(self.n_CIBer)]
        
        sample_weight = self.base_CIBer.sample_weight
        K, nrow = len(self.classes_), len(y_train)
        for i, model in enumerate(self.verb(self.CIBer_lst)) :
            model.sample_weight = sample_weight
            misclassify = y_train != model.predict(X_train)
            err = sum(misclassify*sample_weight)/sum(sample_weight)
            self.beta[i] = np.log((1 - err)/err) + np.log(K-1)
            sample_weight = sample_weight * np.exp(self.beta[i] * misclassify)  # SAMME
            sample_weight = nrow * sample_weight / sum(sample_weight)
        
    def predict_proba(self, X_test):
        y_val = 0
        for i, model in enumerate(self.CIBer_lst):
            y_val = y_val + self.beta[i] * model.predict_proba(X_test)
        
        return np.array(y_val) / np.sum(y_val, axis=1).reshape(-1, 1)
    
#%%
class GradientBoostClassifier():
    def __init__(self, reg, n_reg=100, seed=4012, verbose=True):
        self.reg = reg
        self.n_reg, self.seed = int(n_reg), int(seed)
        self.iter = trange(self.n_reg) if verbose else range(self.n_reg)
        
        self.gamma = np.zeros(self.n_reg)
        self.reg_lst = []
    
    @staticmethod
    def loss(y, p):                                                 # cross-entropy
        return -np.sum(y * np.log(p), axis=1)                       # l_n = sum_i (-y_{ni} ln p_{ni})
    
    @staticmethod
    def gradient(y, p):
        return p - y                                                # g_{ni} = p_{ni} - y_{ni}
    
    @staticmethod
    def hessian(y, p):
        h = - p[:,None,:] * p[:,:,None]                             # h_{nij} = - p_{ni} p_{nj}
        return h + np.einsum('ij,jk->ijk', p, np.eye(p.shape[1]))   # h_{nii} = p_{ni} (1 - p_{ni}) = p_{ni} - p_{ni} p_{ni}
    
    @staticmethod
    def softmax(val):
        exp_val = np.exp(val)
        return exp_val / exp_val.sum(axis=1).reshape(-1, 1)
    
    def fit(self, X_train, y_train, sample_weight=None):
        self.classes_, counts = np.unique(y_train, return_counts=True)
        y_lab = np.eye(len(self.classes_))[y_train.reshape(-1)]
        nrow, self.ncate = y_lab.shape
        
        np.random.seed(self.seed)
        y_val = np.ones(y_lab.shape) * counts / (nrow - counts) + np.random.randint(-5, 5+1, size=y_lab.shape)
        h_val = np.zeros(y_lab.shape)
        for m in self.iter:
            y_prob = self.softmax(y_val)
            neg_grad = -self.gradient(y_lab, y_prob)
            reg_lst = []
            for k in range(self.ncate):
                reg_k = copy_obj(self.reg)
                reg_k.fit(X_train, neg_grad[:,k], sample_weight)
                h_val[:,k] = reg_k.predict(X_train)
            
                reg_lst.append(reg_k)
                
            self.reg_lst.append(reg_lst)
            # gamma_m = sum_n (-g_n.T @ h_n) / sum_n (h_n.T @ H_n @ h_n)
            self.gamma[m] = np.sum(neg_grad * h_val) / np.sum(np.sum(h_val[:,:,None] * self.hessian(y_lab, y_prob), axis=1) * h_val)
            y_val = y_val + self.gamma[m] * h_val
    
    def predict_proba(self, X_test):
        y_val = np.zeros((len(X_test), self.ncate))
        for gamma, model_lst in zip(self.gamma, self.reg_lst):
            for k, reg_k in enumerate(model_lst):
                y_val[:,k] += gamma * reg_k.predict(X_test)
            
        return self.softmax(y_val)
    
    def predict(self, X_test):
        y_proba = self.predict_proba(X_test)
        return self.classes_[list(np.argmax(y_proba, axis=1))]