import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from numba import njit
from typing import Tuple

class ALS:
    def __init__(self, lmbda:float, k:int, n_epochs:int) -> None:
        self.lmbda = lmbda
        self.k = k
        self.n_epochs = n_epochs


    def index_matrix(self, matrix:np.ndarray) -> np.ndarray:
        # Index matrix for training data
        I = matrix.copy()
        I[I > 0] = 1
        I[I == 0] = 0
        return I
    
    # Calculate the RMSE
    def rmse(self, I:np.ndarray,R:np.ndarray,Q:np.ndarray,P:np.ndarray) -> float:
        """RMSE computation

        Args:
            I (np.ndarray)
            R (np.ndarray)
            Q (np.ndarray)
            P (np.ndarray)

        Returns:
            float: RMSE
        """
        return np.sqrt(np.sum((I * (R - np.dot(P.T,Q)))**2)/len(R[R > 0]))
    
    def train(self,train_matrix:np.ndarray, val_matrix:np.ndarray, n,m) -> None:
        """Train the model"""
        R = train_matrix.toarray()
        T = val_matrix.toarray()
        I = self.index_matrix(R)
        I2 = self.index_matrix(T)
        P = 3 * np.random.rand(self.k,m) # Latent user feature matrix
        Q = 3 * np.random.rand(self.k,n) # Latent movie feature matrix
        Q[0,:] = R[R != 0].mean(axis=0) # Avg. rating for each movie
        E = np.eye(self.k) # (k x k)-dimensional idendity matrix

        # First, re-initialize P and Q
        P = 3 * np.random.rand(self.k,m) # Latent user feature matrix
        Q = 3 * np.random.rand(self.k,n) # Latent movie feature matrix
        Q[0,:] = R[R != 0].mean(axis=0) # Avg. rating for each movie

        # Uset different train and test errors arrays so I can plot both versions later
        train_errors_fast = []
        test_errors_fast = []

        # time

        P, Q, train_errors_fast, test_errors_fast = self.als_fast(P, Q, E, R, I, I2, T, self.n_epochs, self.k, self.lmbda, verbose=True)
        print("Algorithm converged")
        self.P = P
        self.Q = Q
        self.train_errors_fast = train_errors_fast
        self.test_errors_fast = test_errors_fast
    # Repeat until convergence
    # @njit
    def fast_step(self, P:np.ndarray, Q:np.ndarray, E:np.ndarray, R:np.ndarray, I:np.ndarray, n_epochs:int, k:int, lmbda:float, verbose=False) -> Tuple[np.ndarray, np.ndarray]:
        """Alternative Least Squares : 1 Step

        Args:
            P (np.ndarray): Latent user feature matrix
            Q (np.ndarray): Latent movie feature matrix
            E (np.ndarray): _description_
            R (np.ndarray): Training matrix
            I (np.ndarray): Index matrix for training data
            n_epochs (int)
            k (int): latent features number
            lmbda (float): regularization parameter
            verbose (bool, optional): Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: P and Q updated
        """
        # Fix Q and estimate P
        for i, Ii in enumerate(I):
            nui = np.count_nonzero(Ii) # Number of items user i has rated
            if (nui == 0): nui = 1 # Be aware of zero counts!
        
            # Least squares solution
            
            #-------------------------------------------------------------------
            # Get array of nonzero indices in row Ii
            Ii_nonzero = np.nonzero(Ii)[0]
            # Select subset of Q associated with movies reviewed by user i
            Q_Ii = Q[:, Ii_nonzero]
            # Select subset of row R_i associated with movies reviewed by user i
            R_Ii = R[i, Ii_nonzero]
            Ai = np.dot(Q_Ii, Q_Ii.T) + lmbda * nui * E
            Vi = np.dot(Q_Ii, R_Ii.T)
            #-------------------------------------------------------------------
            
            P[:, i] = np.linalg.solve(Ai, Vi)
            
        # Fix P and estimate Q
        for j, Ij in enumerate(I.T):
            nmj = np.count_nonzero(Ij) # Number of users that rated item j
            if (nmj == 0): nmj = 1 # Be aware of zero counts!
            
            # Least squares solution
            
            #-----------------------------------------------------------------------
            # Get array of nonzero indices in row Ij
            Ij_nonzero = np.nonzero(Ij)[0]
            # Select subset of P associated with users who reviewed movie j
            P_Ij = P[:, Ij_nonzero]
            # Select subset of column R_j associated with users who reviewed movie j
            R_Ij = R[Ij_nonzero, j]
            Aj = np.dot(P_Ij, P_Ij.T) + lmbda * nmj * E
            Vj = np.dot(P_Ij, R_Ij)
            #-----------------------------------------------------------------------
            
            Q[:,j] = np.linalg.solve(Aj,Vj)

        return P, Q
            
            
    def als_fast(self, P:np.ndarray, Q:np.ndarray, E:np.ndarray, R:np.ndarray, I:np.ndarray, I2:np.ndarray, T:np.ndarray, n_epochs:int, k:int, lmbda:float, verbose=False, early_stopping=False, patience=2) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """Alternative Least Squares : Fast version
        
        Args:
            P (np.ndarray): Latent user feature matrix
            Q (np.ndarray): Latent movie feature matrix
            E (np.ndarray): _description_
            R (np.ndarray): Training matrix
            I (np.ndarray): Index matrix for training data
            I2 (np.ndarray): Index matrix for test data
            T (np.ndarray): Test matrix
            n_epochs (int)
            k (int): latent features number
            lmbda (float): regularization parameter
            verbose (bool, optional): Defaults to False.
            early_stopping (bool, optional): Defaults to False.
            patience (int, optional): Defaults to 2.
            
            Returns:
                Tuple[np.ndarray, np.ndarray, list, list]: P and Q updated, train and test errors
        """
        train_errors_fast = []
        test_errors_fast = []
        if early_stopping:
            new_patience = patience
        # Repeat until convergence
        for epoch in range(n_epochs):
            P, Q = self.fast_step(P, Q, E, R, I, n_epochs, k, lmbda, verbose=False)
            train_rmse = self.rmse(I,R,Q,P)
            test_rmse = self.rmse(I2,T,Q,P)
            train_errors_fast.append(train_rmse)
            test_errors_fast.append(test_rmse)
            if verbose:
                print("[Epoch %d/%d] train error: %f, test error: %f" \
            %(epoch+1, n_epochs, train_rmse, test_rmse))
            if early_stopping:
                if epoch > 1:
                    if test_errors_fast[-1] >= test_errors_fast[-2]:
                        new_patience -= 1
                        if new_patience == 0:
                            print("Early stopping at epoch %d" % epoch)
                            break
                    else:
                        new_patience = patience
        return P, Q, train_errors_fast, test_errors_fast
    
    def save_model(self) -> None:
        np.save("models/ALS/P", self.P)
        np.save("models/ALS/Q", self.Q)
        
    
    def visualisation(self) -> None:
        # Check performance by plotting train and test errors
        # Added curves for errors from updated algorithm to make sure the accuracy is unchanged (aside from random deviations)
        plt.plot(range(self.n_epochs), self.train_errors_fast, marker='o', label='Training Data (Updated)')
        plt.plot(range(self.n_epochs), self.test_errors_fast, marker='v', label='Test Data (Updated)')
        plt.title('ALS-WR Learning Curve')
        plt.xlabel('Number of Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.show()