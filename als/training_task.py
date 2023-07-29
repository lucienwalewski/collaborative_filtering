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
    args_parser.add_argument(
        '--bls_data',
        help="""path to bls dataset""",
        default="../data/prior/bfm/")
    args_parser.add_argument(
        '--use_weightings',
        help="""Whether to use weightings or not""",
        default=False,
        type=bool)

    
    return args_parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    data = Data(train_path=args.dataset, prior_path=args.bls_data)
    model = ALS(lmbda=args.lmbda, k=args.k, n_epochs=args.epochs)
    pipeline = TrainingPipeline(data, model)
    pipeline.execute(use_weights=args.use_weightings)
    print("Model trained and saved")

    
