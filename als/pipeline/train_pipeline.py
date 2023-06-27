from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data_processing import Data
from models import ALS, MLP
from typing import Self, Union


class TrainingPipeline:
    def __init__(self, data:Data, model: Union[ALS, MLP]) -> None:
        """Training pipeline for models
        
        Args:
            data (Data): Data object
            model (Union[ALS, MLP]): Model object
            
        """
        self.data = data
        self.model = model

    def execute(self) -> None:
        train_matrix, val_matrix = self.data.get_matrices()
        n_rows, n_cols = self.data.get_shape()
        self.model.train(train_matrix, val_matrix, n_cols, n_rows)
        self.model.save_model()
    
class ValidationPipeline:
    def __init__(self, data:Data, path="models/ALS") -> None:
        """Validation pipeline for models
        
        Args:
            data (Data): Data object
            P,Q from ALS model
            
        """
        self.P = np.load(path + "/P.py")
        self.Q = np.load(path + "/Q.py")
        self.data = data
        

    def execute(self) -> None:
        self.prediction()
        self.mse()
        self.prediction_analysis = pd.merge(self.data_analysis, self.mse_df,left_index=True, right_index=True, how='inner')
        self.visualize()
    
    def prediction(self) -> None:
        prediction_matrix = np.dot(self.P.T, self.Q)
        # make dataframe out of flatten matrix with features UserId and MovieId
        prediction_df = pd.DataFrame(prediction_matrix.flatten(), columns=['Prediction'])
        prediction_df['UserId'] = np.repeat(np.arange(self.n_rows), self.n_cols)
        prediction_df['MovieId'] = np.tile(np.arange(self.n_cols), self.n_rows)
        prediction_df = prediction_df[['UserId', 'MovieId', 'Prediction']]
        # set idx to format rX_cY
        prediction_df['idx'] = 'r' + prediction_df['UserId'].astype('str') + '_c' + prediction_df['MovieId'].astype('str')
        prediction_df = prediction_df.set_index('idx')
        self.val_data_scope = self.test_data.merge(prediction_df['Prediction'], left_index=True, right_index=True)

    def mse(self) -> None:
        grouped = self.val_data_scope.groupby(by="MovieId")
        mse = grouped.apply(lambda x: np.mean((x['Rating'] - x['Prediction']) ** 2))
        mse_df = pd.DataFrame(mse, columns=['MSE'])

    def visualize(self) -> None:
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        # Plot the data on each subplot
        self.prediction_analysis.sort_values(by="sparsity").plot(x='sparsity', y='MSE', kind='scatter', ax=axs[0])
        axs[0].set_title('MSE as a function of sparsity')
        
        self.prediction_analysis.sort_values(by="grading_avg").plot(x='grading_avg', y='MSE', kind='scatter', ax=axs[1])
        axs[1].set_title('MSE as a function of gradxing_avg')
        
        self.prediction_analysis.sort_values(by="grading_std").plot(x='grading_std', y='MSE', kind='scatter', ax=axs[2])
        axs[2].set_title('MSE as a function of grading_std')
        
        # Adjust the spacing between subplots
        plt.tight_layout()
        # Show the plots
        a = 1
        plt.show()