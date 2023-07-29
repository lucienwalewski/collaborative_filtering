import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from typing import Tuple, Optional

class ALS:
    def __init__(self, lmbda: float, k: int, n_epochs: int) -> None:
        self.lmbda = lmbda
        self.k = k
        self.n_epochs = n_epochs

    def index_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Index matrix for training/test data"""
        I = np.zeros(matrix.shape)
        I[matrix != 0] = 1
        return I
    
    def rmse(self, I: np.ndarray, R: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
        """RMSE computation"""
        non_zeros = np.count_nonzero(I)
        return np.sqrt(np.sum((I * (R - np.dot(U.T, V)))**2)/non_zeros)

    def als_step(self, train: np.ndarray, U: np.ndarray, V: np.ndarray, I: np.ndarray, I_T: np.ndarray, I_V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one epoch of als"""
        # Fix V and estimate U
        for i, I_Ti in enumerate(I_T):
            # Edge case of zero counts
            non_zero_i = np.count_nonzero(I_Ti) if np.count_nonzero(I_Ti) > 0 else 1
            I_Ti_nonzero = np.nonzero(I_Ti)[0] # Non zero inidices in row i
            V_I_Ti = V[:, I_Ti_nonzero] # Subset of V for row i
            A = np.dot(V_I_Ti, V_I_Ti.T) + self.lmbda * non_zero_i * I
            b = np.dot(V_I_Ti, train[i, I_Ti_nonzero].T)
            U[:, i] = np.linalg.solve(A, b)

        # CITATION: The use of np.linalg.solve was inspired by the following blog post:
        # https://github.com/benlindsay/als-wr-tutorial/blob/master/modified_notebook.ipynb

        # Fix U and estimate V
        for j, I_Tj in enumerate(I_T.T):
            # Edge case of zero counts
            non_zero_j = np.count_nonzero(I_Tj) if np.count_nonzero(I_Tj) > 0 else 1
            I_Tj_nonzero = np.nonzero(I_Tj)[0] # Non zero inidices in row j
            U_I_Tj = U[:, I_Tj_nonzero] # Subset of U for row j
            A = np.dot(U_I_Tj, U_I_Tj.T) + self.lmbda * non_zero_j * I
            b = np.dot(U_I_Tj, train[I_Tj_nonzero, j])
            V[:, j] = np.linalg.solve(A, b)
        
        return U, V

    def weighted_als_step(self, train: np.ndarray, U: np.ndarray, V: np.ndarray, I: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one epoch of als given a weight matrix"""
        # Fix V and estimate U
        for i in range(train.shape[0]):
            A = W[i].T * V @ V.T + self.lmbda * np.eye(self.k)
            b = (W[i] * train[i]) @ V.T
            U[:, i] = np.linalg.solve(A, b)

        # Fix U and estimate V
        for j in range(train.shape[1]):
            A = U @ (W[:, j] * U).T + self.lmbda * np.eye(self.k)
            b = U @ (W[:, j] * train[:, j])
            V[:, j] = np.linalg.solve(A, b)
        
        return U, V

    def train(self, train_matrix: csr_matrix, val_matrix: csr_matrix, n: int, m: int, early_stopping: bool=False, patience: int=2, weight_matrix: Optional[np.ndarray]=None) -> None:
        """Train the model"""
        # Initialization
        train = train_matrix.toarray()
        val = val_matrix.toarray()
        I_train = self.index_matrix(train)
        I_val = self.index_matrix(val)
        U = 3 * np.random.rand(self.k, m) # Latent user feature matrix
        V = 3 * np.random.rand(self.k, n) # Latent movie feature matrix
        V[0, :] = train[train != 0].mean(axis=0) # Avg. rating for each movie
        I = np.eye(self.k) # (k x k)-dimensional idendity matrix

        if weight_matrix is not None:
            # Set weights to 1 if 1 in R
            weight_matrix[train > 0] = 1

        # For plotting
        train_errors = []
        val_errors = []

        if early_stopping:
            new_patience = patience
        for epoch in range(self.n_epochs):
            if weight_matrix is not None:
                U, V = self.weighted_als_step(train, U, V, I, weight_matrix)
            else:
                U, V = self.als_step(train, U, V, I, I_train, I_val)
            train_rmse = self.rmse(I_train, train, U, V)
            test_rmse = self.rmse(I_val, val, U, V)
            train_errors.append(train_rmse)
            val_errors.append(test_rmse)
            if early_stopping:
                if test_rmse < min(val_errors):
                    new_patience = patience
                else:
                    new_patience -= 1
                    if new_patience == 0:
                        break
            print(f'[{epoch+1}/{self.n_epochs}] Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')

        print("Algorithm converged")
        self.U = U
        self.V = V
        self.train_errors = train_errors
        self.val_errors = val_errors

    def save_model(self, path: str="models/ALS_weights/") -> None:
        """Save model to path"""
        np.save(f"{path}U.npy", self.U.T)
        np.save(f"{path}V.npy", self.V.T)