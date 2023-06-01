from pipeline import TrainingPipeline
from models import ALS
from data_processing import Data
import numpy as np
import argparse

def get_args():
    args_parser = argparse.ArgumentParser()

    # Experiment arguments
    args_parser.add_argument(
        '--dataset',
        help="""path to dataset""",
        default="../data/data_train.csv")
    args_parser.add_argument(
        '--epochs',
        help="""how many epochs to train for""",
        default=200,
        type=int)
    args_parser.add_argument(
        '--lmbda',
        help="""Regularization parameter""",
        default=0.1,
        type=int)
    args_parser.add_argument(
        '--k',
        help="""Number of latent factors""",
        default=10,
        type=int)
    
    return args_parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    data = Data(args.dataset)
    model = ALS(lmbda=args.lmbda, k=args.k, n_epochs=args.epochs)
    pipeline = TrainingPipeline(data, model)
    pipeline.execute()
    print("Model trained and saved")

    
