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
        '--model',
        help="""mlp or mf or ncf""",
        default="mlp")
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
        default=1024,
        type=int)
    args_parser.add_argument(
        '--normalize-by',
        help="""normalize by movie or user or not""",
        default="")

    # both model arguments
    args_parser.add_argument(
        '--weight-decay',
        help="""used for l2 regularization""",
        default=0.1,
        type=float)
    args_parser.add_argument(
        '--out-dim',
        help="""classification or regression""",
        default=5,
        type=int)

    # mlp model arguments
    args_parser.add_argument(
        "--mlp-out-dim",
        type=int,
        default=32,
        help="predictive factors numbers in the mlp model")
    args_parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="number of layers in MLP model")
    args_parser.add_argument(
        '--dropout',
        help="""used in MLP""",
        default=0.3,
        type=float)

    # mf model arguments
    args_parser.add_argument(
        "--mf-embedding-dim",
        type=int,
        default=10,
        help="predictive factors numbers in the mf model")

    # init arguments
    args_parser.add_argument(
        '--mf-pretrained',
        help="""path to pretrained mf model""",
        default="")
    args_parser.add_argument(
        '--mlp-pretrained',
        help="""path to pretrained mlp model""",
        default="")

    return args_parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)

    # dataset
    train_data, val_data, user_num, movie_num = load_cil(args.dataset)
    train_dataset = MLPDataset(train_data, user_num, movie_num, normalize_by=args.normalize_by)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=5)

    val_loader = None
    if val_data is not None:
        val_dataset = MLPDataset(val_data, user_num, movie_num,
                                 mean=train_dataset.mean, std=train_dataset.std, normalize_by=args.normalize_by)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5)

    mean = std = None
    if args.normalize_by != "":
        mean = torch.tensor(train_dataset.mean.values).float()
        std = torch.tensor(train_dataset.std.values).float()

    # model
    mlp_model = MLPModel(args, user_num, movie_num, mean, std)

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

