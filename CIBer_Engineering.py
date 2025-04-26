import os, sys
from contextlib import contextmanager
import numpy as np
import pandas as pd
from scipy import stats
#from mdlp import MDLP

@contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    yield
    sys.stdout = save_stdout

SCIPY_DIST = ["uniform", "norm", "t", "chi2", "expon", "laplace", "skewnorm", "gamma"]
SIZE_BASE = ["equal_size", "pkid", "ndd", "wpkid"]
DISC_BASE = ["mdlp", "equal_length", "auto"] + SIZE_BASE
MDLP_BACKUP = ["equal_length", "auto"] + SIZE_BASE

class Disc_Layer():
    def __init__(self, cont_col, disc_method, n_bins=50):
        self.cont_col = cont_col
        self.disc_method = disc_method.lower()
        self.n_bins = int(n_bins)
        
        assert self.n_bins > 1
    
    def transform(self, X_test, y_test=None):
        discret_X_test = X_test.copy()
        for col in self.cont_col:
            discret_X_test[:,col] = np.digitize(X_test[:,col], self.bins_discret[col], right=True)
            
        return discret_X_test

class disc_distribution(Disc_Layer):
    def __init__(self, cont_col, disc_method="norm", n_bins=50, **kwargs):
        super().__init__(cont_col, disc_method, n_bins)
        
        assert self.disc_method in SCIPY_DIST + ["auto"]
    
    def _distribution_discretize(self, feature):
        if self.disc_method == "auto":
            distribution = self._select_distribution(feature)
        else:
            distribution = getattr(stats, self.disc_method)
            
        params = distribution.fit(feature)
        bins = distribution.ppf(np.arange(1, self.n_bins)/self.n_bins, *params)
        return np.digitize(feature, bins, right=True), bins
    
    def fit_transform(self, X_train, y_train=None):
        self.bins_discret = dict()
        discret_X_train = X_train.copy()
        for col in self.cont_col:
            discret_X_train[:,col], self.bins_discret[col] = self._distribution_discretize(X_train[:,col])
            
        return discret_X_train

    def _select_distribution(self, feature):
        best_ks, best_disc = np.inf, None
        for disc_method in SCIPY_DIST:
            ks, distribution = self._KS_test(feature, disc_method)
            if ks < best_ks:
                best_disc = distribution
                best_ks = ks
                
        return best_disc
    
    @staticmethod
    def _KS_test(feature, disc_method):
        distribution = getattr(stats, disc_method)
        try:
            params = distribution.fit(feature)
            ks, _ = stats.kstest(feature, distribution.cdf, args=params)
        except:
            ks = np.inf
            
        return ks, distribution

class disc_equal_size(Disc_Layer):
    def __init__(self, cont_col, disc_method="equal_size", n_bins=50, n=3, m=30, **kwargs):
        super().__init__(cont_col, disc_method, n_bins)
        self.n, self.m = int(n), int(m)
        
        assert self.disc_method in SIZE_BASE
        assert self.m > 1 and self.n > 0
        assert self.n % 2 == 1
    
    def _equal_size_discretize(self, feature):
        # If some b_i's are the same, then we will skip some discretized numbers
        bins = np.sort(feature)[(self.bin_size-1)::self.bin_size]
        return np.digitize(feature, bins, right=True), bins
    
    def fit_transform(self, X_train, y_train=None):
        N_train = np.shape(X_train)[0]
        if self.disc_method == "pkid":
            self.n_bins = np.floor(np.sqrt(N_train))
        elif self.disc_method == "ndd":
            #self.n_bins = np.floor(np.sqrt(N_train) * self.n) # Original Paper
            self.n_bins = np.floor(np.sqrt(N_train) / self.n) # IDK why this is better
        elif self.disc_method == "wpkid":
            # t^2 + mt - N = 0
            self.n_bins = (-self.m + np.sqrt(self.m**2 + 4*1*N_train))/(2*1)
        
        self.bin_size = int(N_train // self.n_bins + 1)
        self.bins_discret = dict()
        discret_X_train = X_train.copy()
        for col in self.cont_col:
            discret_X_train[:,col], self.bins_discret[col] = self._equal_size_discretize(X_train[:,col])
            
        return discret_X_train

class disc_mdlp(Disc_Layer):
    def __init__(self, cont_col, disc_backup="pkid", n_bins=50, **kwargs):
        super().__init__(cont_col, "mdlp", n_bins)
        self.disc_backup = disc_backup.lower()
        
        assert self.disc_backup in MDLP_BACKUP + SCIPY_DIST
    
    def fit_transform(self, X_train, y_train):
        cate_col = set(np.arange(np.shape(X_train)[1])) - set(self.cont_col)
        self.discretize = MDLP(categorical_features=cate_col)
        
        with nostdout():
            self.discretize.fit(X_train, y_train)
            
        self.bins_discret = self.discretize.cut_points_
        fail_col = [col for col in self.cont_col if len(self.bins_discret[col]) == 0]
        if len(fail_col) > 0:
            print(f"MDLP fails, using {self.disc_backup}-discretization with {self.n_bins} bins in column(s) {fail_col} instead of MDLP")
        
        if len(fail_col) > 0:
            if self.disc_backup in SIZE_BASE:
                self.base_discretizer = disc_equal_size(fail_col, disc_method=self.disc_backup, n_bins=self.n_bins)
            elif self.disc_backup in SCIPY_DIST:
                self.base_discretizer = disc_distribution(fail_col, disc_method=self.disc_backup, n_bins=self.n_bins)
            elif self.disc_backup == "equal_length":
                self.base_discretizer = disc_distribution(fail_col, disc_method="uniform", n_bins=self.n_bins)
            
            self.base_discretizer.fit_transform(X_train)
            self.bins_discret.update(self.base_discretizer.bins_discret)
        
        return self.transform(X_train)
    
def Discretization(cont_col, disc_method, **kwargs):
    disc_method = disc_method.lower()
    if disc_method == "equal_length":
        disc_method = "uniform"
    assert disc_method in DISC_BASE + SCIPY_DIST
        
    if disc_method in SIZE_BASE:
        return disc_equal_size(cont_col, disc_method=disc_method, **kwargs)
    elif disc_method in SCIPY_DIST + ["auto"]:
        return disc_distribution(cont_col, disc_method=disc_method, **kwargs)
    elif disc_method == "mdlp":
        return disc_mdlp(cont_col, **kwargs)

#%%
class Joint_Encoding():
    def _contingency_table(self, df):
        col_index = list(df.columns.values)
        return (df.groupby(col_index, as_index=False).size()
                .sort_values(['size'] + col_index, ascending=[False] + [True]*len(col_index))
                .reset_index(drop=True)
                .drop(['size'], axis=1))
    
    def _encoding(self, df):
        columns, encode_num = df.columns.values, 0
        col_order = np.argsort(df.nunique(axis=0).values)[::-1]                         # decreasing col order with number of categories
        
        frequency_df = self._contingency_table(df.copy())
        encode_ref = {k: dict.fromkeys(frequency_df[k].unique()) for k in columns}      # encode number {feature: {value: encode_num}}
        
        while len(columns) > 0:
            encoded = False
            for row in frequency_df.itertuples(index=False):
                # if np.all([encode_ref[col][val] is None for col, val in zip(columns, row)]): # slightly faster
                encode_check = True
                for col, val in zip(columns, row):
                    if encode_ref[col][val] is not None:
                        encode_check = False
                        break
                
                if encode_check:
                    for col, val in zip(columns, row):
                        encode_ref[col][val] = encode_num
                    encode_num += 1
                    encoded = True
                    
                    if encode_num == len(encode_ref[col_order[-1]])-1:
                        break                           # break one for-loop
            
            while len(columns) > 0 and encode_num == len(encode_ref[col_order[-1]])-1:
                for index in encode_ref[col_order[-1]]:
                    if encode_ref[col_order[-1]][index] is None:
                        encode_ref[col_order[-1]][index] = encode_num
                        break                           # break one for-loop
                
                df.drop(col_order[-1], axis=1, inplace=True)
                columns, col_order = df.columns.values, col_order[:-1]
            
            # run through all rows but none of the categories are encoded
            if encoded is False:
                # assign random numbers to the remaining categories (actually, preordered by frequency_df)
                for col in columns:
                    m = encode_num
                    for index in encode_ref[col]:
                        if encode_ref[col][index] is None:
                            encode_ref[col][index] = m
                            m += 1
                
                return encode_ref
            
            if len(columns) > 0:
                frequency_df = self._contingency_table(df.copy())

        return encode_ref
    
    def fit_transform(self, X_train, y_train=None):
        self.col_index = list(np.arange(len(self.cate_col)))
        df_train = pd.DataFrame(X_train[:,self.cate_col], columns=self.col_index)
        self.encode_ref = self._encoding(df_train.copy())
        X_train[:,self.cate_col] = df_train.replace(self.encode_ref).to_numpy()
        return X_train
    
    def transform(self, X_test, y_test=None):
        df_test = pd.DataFrame(X_test[:,self.cate_col], columns=self.col_index)
        for col in self.col_index:
            for item in list(set(df_test[col].unique()) - set(self.encode_ref[col])):
                self.encode_ref[col][item] = len(self.encode_ref[col]) + 1
                #print(f"Add encoding: Feature: {col} and Item: {item} with value: {len(self.encode_ref[col])}")
        
        X_test[:,self.cate_col] = df_test.replace(self.encode_ref).to_numpy()
        return X_test