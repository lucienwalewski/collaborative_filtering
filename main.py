from Pipeline import TrainingPipeline
from models import ALS, MLP
from data_processing import Data

if __name__ == "__main__":
    data = Data("data/data_train.csv")
    model = ALS(lmbda=0.1, k=10, n_epochs=1)
    pipeline = TrainingPipeline(data, model)
    pipeline.execute()
    print("Model trained and saved")