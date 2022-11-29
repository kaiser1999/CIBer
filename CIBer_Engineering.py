import os, sys
from contextlib import contextmanager
import numpy as np
import pandas as pd
from scipy import stats
from mdlp import MDLP
import concurrent.futures

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

class disc_distribution():
    def __init__(self, cont_col, disc_method="norm", n_bins=50, **kwargs):
        self.cont_col = cont_col
        self.disc_method = disc_method.lower()
        self.n_bins = int(n_bins)
        
        assert self.disc_method in SCIPY_DIST + ["auto"]
        assert self.n_bins > 1
    
    def _distribution_discretize(self, feature):
        if self.disc_method == "auto":
            distribution = self._select_distribution(feature)
        else:
            distribution = getattr(stats, self.disc_method)
        params = distribution.fit(feature)
        bins = distribution.ppf(np.arange(1, self.n_bins)/self.n_bins, *params)
        return np.digitize(feature, bins, right=True), bins
    
    def fit_transform(self, x_train, y_train=None):
        self.bins_discret = dict()
        discret_x_train = x_train.copy()
        for col in self.cont_col:
            discret_x_train[:,col], self.bins_discret[col] = self._distribution_discretize(x_train[:,col])
        return discret_x_train
    
    def transform(self, x_test):
        discret_x_test = x_test.copy()
        for col in self.cont_col:
            discret_x_test[:,col] = np.digitize(x_test[:,col], self.bins_discret[col], right=True)
        return discret_x_test

    def _select_distribution(self, feature):
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            KS_results = [executor.submit(self._KS_test, feature, disc_method).result() for disc_method in SCIPY_DIST]
        
        best_ks, best_disc = np.inf, None
        for ks, distribution in KS_results:
            if ks < best_ks:
                best_disc = distribution
                best_ks = ks
        return best_disc
    
    def _KS_test(self, feature, disc_method):
        distribution = getattr(stats, disc_method)
        try:
            params = distribution.fit(feature)
            ks, _ = stats.kstest(feature, distribution.cdf, args=params)
        except:
            ks = np.inf
        return ks, distribution

class disc_equal_size():
    def __init__(self, cont_col, disc_method="equal_size", n_bins=50, n=3, m=30, **kwargs):
        self.cont_col = cont_col
        self.disc_method = disc_method
        self.n_bins = int(n_bins)
        self.n = int(n)
        self.m = int(m)
        
        assert self.disc_method in SIZE_BASE
        assert self.n_bins > 1 and self.m > 1 and self.n > 0
        assert self.n % 2 == 1
    
    def _equal_size_discretize(self, feature):
        # If some b_i's are the same, then we will skip some discretized numbers
        split = np.array_split(np.sort(feature), self.n_bins)
        bins = [x[-1] for x in split]
        bins = bins[:-1]
        return np.digitize(feature, bins, right=True), bins
    
    def fit_transform(self, x_train, y_train=None):
        N_train = np.shape(x_train)[0]
        if self.disc_method == "pkid":
            self.n_bins = np.floor(np.sqrt(N_train))
        elif self.disc_method == "ndd":
            #self.n_bins = np.floor(np.sqrt(N_train) * self.n) # Original Paper
            self.n_bins = np.floor(np.sqrt(N_train) / self.n) # IDK why this is better
        elif self.disc_method == "wpkid":
            # t^2 + mt - N = 0
            self.n_bins = (-self.m + np.sqrt(self.m**2 + 4*1*N_train))/(2*1)
        
        self.bins_discret = dict()
        discret_x_train = x_train.copy()
        for col in self.cont_col:
            discret_x_train[:,col], self.bins_discret[col] = self._equal_size_discretize(x_train[:,col])
        return discret_x_train
    
    def transform(self, x_test):
        discret_x_test = x_test.copy()
        for col in self.cont_col:
            discret_x_test[:,col] = np.digitize(x_test[:,col], self.bins_discret[col], right=True)
        return discret_x_test

class disc_mdlp():
    def __init__(self, cont_col, disc_backup="pkid", n_bins=50, **kwargs):
        self.cont_col = cont_col
        self.disc_backup = disc_backup.lower()
        assert self.disc_backup in MDLP_BACKUP + SCIPY_DIST
        self.n_bins = int(n_bins)
        assert n_bins > 1
    
    def fit_transform(self, x_train, y_train):
        cate_col = set(np.arange(np.shape(x_train)[1])) - set(self.cont_col)
        self.discretize = MDLP(categorical_features=cate_col)
        
        with nostdout():
            self.discretize.fit(x_train, y_train)
            
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
            
            self.base_discretizer.fit_transform(x_train)
            self.bins_discret.update(self.base_discretizer.bins_discret)
        
        return self.transform(x_train)
    
    def transform(self, x_test):
        discret_x_test = x_test.copy()
        for col in self.cont_col:
            discret_x_test[:,col] = np.digitize(x_test[:,col], self.bins_discret[col], right=True)
        return discret_x_test
    
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

class Joint_Encoding():
    def _contingency_table(self, df, col_index):
        frequency_df = df.groupby(col_index).size().reset_index(name='count')
        return frequency_df.sort_values(by=['count'], ascending=False).reset_index(drop=True).drop(['count'], axis=1)
        
    def fit(self, x_train):
        col_index = list(np.arange(np.shape(x_train)[1]))
        df = pd.DataFrame(x_train, columns=col_index)
        
        frequency_df = self._contingency_table(df, col_index)
        self.col_order = np.argsort([len(df[k].unique()) for k in col_index])[::-1]
        
        encode_ref = {k:{v:None for v in frequency_df[k].unique()} for k in col_index} # encode number {feature:{value:encoded num}}
        reduced_df = df[self.col_order].copy()
        
        encode_num = 0
        encode_ref = self._encoding(reduced_df, frequency_df, encode_ref, encode_num)
        
        for col in encode_ref.keys():
            df[col].replace(encode_ref[col], inplace=True)

        self.encode_ref = encode_ref
        self.col_index = col_index
        return df.to_numpy()
    
    def _encoding(self, reduced_df, frequency_df, encode_ref, encode_num):
        nrow, ncol = frequency_df.shape
        columns = reduced_df.columns.values
        for row in range(nrow):
            # If none of the columns are encoded
            if np.all([encode_ref[col][frequency_df[col].iloc[row]] is None for col in columns]):
                for col in columns:
                    encode_ref[col][frequency_df[col].iloc[row]] = encode_num
                encode_num += 1
        if len(reduced_df.columns) > 1 and encode_num < len(encode_ref[self.col_order[ncol-1]]):
            m = encode_num
            for index in encode_ref[self.col_order[ncol-1]]:
                if encode_ref[self.col_order[ncol-1]][index] is None:
                    encode_ref[self.col_order[ncol-1]][index] = m
                    m += 1
        reduced_df.drop(reduced_df.columns[ncol-1], axis=1, inplace=True)
        
        if len(reduced_df.columns) > 0:
            frequency_df = self._contingency_table(reduced_df, list(reduced_df.columns.values))
            return self._encoding(reduced_df, frequency_df, encode_ref, encode_num)
        else:
            return encode_ref
    
    def transform(self, x_test):
        df = pd.DataFrame(x_test, columns=self.col_index)
        
        for col in self.col_index:
            for item in list(set(df[col].unique()) - set(self.encode_ref[col])):
                self.encode_ref[col][item] = len(self.encode_ref[col]) + 1
                #print(f"Add encoding: Feature: {col} and Item: {item} with value: {len(self.encode_ref[col])}")
            
        for col in self.encode_ref.keys():
            df[col].replace(self.encode_ref[col], inplace=True)
            
        return df.to_numpy()

class Frequency_Encoding():
    def _contingency_table(self, df, col_index):
        frequency_df = df.groupby(col_index).size().reset_index(name='count')
        return frequency_df.sort_values(by=['count'], ascending=False).reset_index(drop=True).drop(['count'], axis=1)
        
    def fit(self, x_train):
        self.encode_ref = dict()
        self.col_index = list(np.arange(np.shape(x_train)[1]))
        df = pd.DataFrame(x_train, columns=self.col_index)
        self.non_dummy_col = [col for col in df.columns if len(df[col].unique()) != 2]
        
        frequency_df = self._contingency_table(df, self.non_dummy_col)
        for col in self.non_dummy_col:
            encode = pd.Series(frequency_df[col].values).unique()
            self.encode_ref[col] = dict(zip(encode, np.arange(len(encode))))
            df[col].replace(self.encode_ref[col], inplace=True)

        return df.to_numpy()
    
    def transform(self, x_test):
        df = pd.DataFrame(x_test, columns=self.col_index)
        
        for col in self.non_dummy_col:
            for item in list(set(df[col].unique()) - set(self.encode_ref[col])):
                self.encode_ref[col][item] = len(self.encode_ref[col]) + 1
                #print(f"Add encoding: Feature: {col} and Item: {item} with value: {len(self.encode_ref[col])}")
            
        for col in self.encode_ref.keys():
            df[col].replace(self.encode_ref[col], inplace=True)
            
        return df.to_numpy()