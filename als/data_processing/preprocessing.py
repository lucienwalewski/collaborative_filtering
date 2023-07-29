import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Tuple

class Data:
    def __init__(self, train_path: str, prior_path: str) -> None:
        self.data = pd.read_csv(train_path, index_col=0)
        self.prior_path = prior_path
        self.preprocess()
        self.load_bfm()
        self.calculate_weightings()
        self.split_train_test()
        self.get_sparse_matrix()
        self.analytics()

    def preprocess(self) -> None:
        # rename column and turn ot uint8
        self.data.rename(columns={'Prediction': 'Rating'}, inplace=True)
        self.data['Rating'] = self.data['Rating'].astype('uint8')

        # get user and movie id by splitting index given in format rX_cY
        self.data['UserId'] = self.data.index.str.split('_').str[0].str[1:].astype('int32')
        self.data['MovieId'] = self.data.index.str.split('_').str[1].str[1:].astype('int32')

        # subtract min UserId and MovieID to get indices starting at 0
        self.data['UserId'] = self.data['UserId'] - self.data['UserId'].min()
        self.data['MovieId'] = self.data['MovieId'] - self.data['MovieId'].min()

        # reorder columns to UserId, MovieId, Rating
        self.data = self.data[['UserId', 'MovieId', 'Rating']]

    def split_train_test(self, test_size: float=0.2):
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=42)

    def construct_sparse_matrix(self, data:pd.DataFrame, n_rows:int, n_cols:int) -> csr_matrix:
        return csr_matrix((data['Rating'].values, (data['UserId'].values, data['MovieId'].values)), shape=(n_rows, n_cols))
    
    def get_sparse_matrix(self) -> None:
        self.n_rows = self.train_data['UserId'].max() + 1
        self.n_cols = self.train_data['MovieId'].max() + 1
        self.train_matrix = self.construct_sparse_matrix(self.train_data, self.n_rows, self.n_cols)
        self.val_matrix = self.construct_sparse_matrix(self.test_data, self.n_rows, self.n_cols)

    def get_matrices(self) -> Tuple[csr_matrix, csr_matrix]:
        return self.train_matrix, self.val_matrix
    
    def get_shape(self) -> Tuple[int, int]:
        return (self.n_rows, self.n_cols)
    
    def analytics(self) -> None:
        sparsity = (self.n_cols - self.train_data['MovieId'].value_counts())/self.n_cols
        average_per_movie = self.train_data.groupby(by="MovieId")['Rating'].mean()
        median_per_movie = self.train_data.groupby(by="MovieId")['Rating'].median()
        train_std = self.train_data.groupby(by="MovieId")['Rating'].std()
        self.train_data_analysis = pd.DataFrame({'sparsity':sparsity, 'grading_avg':average_per_movie,'grading_median':median_per_movie,'grading_std':train_std})

    def load_bfm(self) -> None:
        # Load prior data as np arrays into a dict
        prior = {}
        prior['movie_bias'] = np.load(self.prior_path + "movie_bias.npy")
        prior['user_bias'] = np.load(self.prior_path + "user_bias.npy")
        prior['U'] = np.load(self.prior_path + "U.npy")
        prior['V'] = np.load(self.prior_path + "V.npy")
        prior['w0'] = np.load(self.prior_path + "w0.npy")
        self.prior = prior

    def calculate_weightings(self) -> None:
        # Calculate the weightings by first getting the standard deviation of the ratings from bfm
        preds = np.zeros((200, 10000, 1000))
        for i in range(200):
            preds[i] = self.prior['U'][i] @ self.prior['V'][i].T + self.prior['w0'][i] + self.prior['movie_bias'][i] + self.prior['user_bias'][i][:, np.newaxis]
        std = np.std(preds, axis=0)
        # Then calculate the weightings
        self.weightings = 1 / (1 + np.exp(5 * std))
        # Replace observed values with weights of 1
        # FIXME