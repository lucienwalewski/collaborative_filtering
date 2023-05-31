from data_processing import Data
from models import ALS, MLP
from typing import Union

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
        self.model.fit(train_matrix, val_matrix)
        self.model.save_model()
    

    