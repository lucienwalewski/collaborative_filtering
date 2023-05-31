from Pipeline import TrainingPipeline
from models import ALS, MLP
from data_processing import Data

if __name__ == "__main__":
    data = Data("data/data_train.csv")
    model = ALS()
    pipeline = TrainingPipeline(data, model)
    pipeline.execute()