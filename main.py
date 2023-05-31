from Pipeline import TrainingPipeline
from models import ALS, MLP
from data_processing import Data

if __name__ == "__main__":
    data = Data("data/data_train.csv")
    model = ALS(lmbda=0.1, k=10, nb_users=10000, nb_items=1000, n_epochs=150)
    pipeline = TrainingPipeline(data, model)
    pipeline.execute()
    print("Model trained and saved")