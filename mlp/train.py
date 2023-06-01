import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping

from utils.dataset import MLPDataset
from utils.model import MLPModel
from utils.helper import load_cil

import argparse


def get_args():
    args_parser = argparse.ArgumentParser()

    # Experiment arguments
    args_parser.add_argument(
        '--dataset',
        help="""set to train or split""",
        default="split")
    args_parser.add_argument(
        '--epochs',
        help="""how many epochs to train for""",
        default=200,
        type=int)
    args_parser.add_argument(
        '--lr',
        help='Learning rate value for the optimizers.',
        default=1e-4,
        type=float)
    args_parser.add_argument(
        '--batch-size',
        help="""for train and val loader""",
        default=512,
        type=int)

    # model arguments
    args_parser.add_argument(
        "--factor-num",
        type=int,
        default=32,
        help="predictive factors numbers in the model")
    args_parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="number of layers in MLP model")
    args_parser.add_argument(
        '--weight-decay',
        help="""used for l2 regularization""",
        default=0.1,
        type=float)
    args_parser.add_argument(
        '--dropout',
        help="""used in MLP""",
        default=0.3,
        type=float)

    return args_parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)

    # dataset
    train_data, val_data, user_num, movie_num = load_cil(args.dataset)
    train_dataset = MLPDataset(train_data, user_num, movie_num)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=5)

    val_loader = None
    if val_data is not None:
        val_dataset = MLPDataset(val_data, user_num, movie_num)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5)

    # model
    mlp_model = MLPModel(args, user_num, movie_num)

    callbacks = [TQDMProgressBar(refresh_rate=200)]
    if val_data is not None:
        callbacks.append(ModelCheckpoint(monitor='val_rmse', save_top_k=1, mode='min'))
        callbacks.append(EarlyStopping(monitor='val_rmse', patience=10, mode='min'))

    # train
    training_args = {
        "max_epochs": args.epochs,
        "callbacks": callbacks,
        "accelerator": 'cuda' if torch.cuda.is_available() else 'mps',
        "devices": 1,
    }

    trainer = Trainer(**training_args)
    trainer.fit(mlp_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Finished training")

