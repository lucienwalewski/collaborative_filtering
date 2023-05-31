import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class Data:
    def __init__(self, path:str) -> None:
        self.data = pd.read_csv(path, index_col=0)
        self.preprocess()
        self.split_train_test()
        self.get_sparse_matrix()

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

    def split_train_test(self):
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)

    def construct_sparse_matrix(self, data:pd.DataFrame, n_rows:int, n_cols:int) -> csr_matrix:
        return csr_matrix((data['Rating'].values, (data['UserId'].values, data['MovieId'].values)), shape=(n_rows, n_cols))
    
    def get_sparse_matrix(self) -> None:
        n_rows = self.train_data['UserId'].max() + 1
        n_cols = self.train_data['MovieId'].max() + 1
        self.train_matrix = self.construct_sparse_matrix(self.train_data, n_rows, n_cols)
        self.val_matrix = self.construct_sparse_matrix(self.test_data, n_rows, n_cols)