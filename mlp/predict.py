import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from utils.dataset import MLPDataset
from utils.model import MLPModel
from utils.helper import load_cil

from torchmetrics import MeanSquaredError

if __name__ == '__main__':

    version = "mlp_new_mse_val"
    dataset = "val"

    # dataset
    train_data, val_data, user_num, movie_num = load_cil(dataset=dataset)

    # train normalization
    normalize_by = ""
    mean = std = None
    if normalize_by != "":
        mean = train_data.groupby(by=normalize_by)['rating'].mean()
        std = train_data.groupby(by=normalize_by)['rating'].std()

    val_dataset = MLPDataset(val_data, user_num, movie_num, mean, std, normalize_by)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=5)

    # get filename of best checkpoint
    path = f"lightning_logs/{version}/checkpoints/"
    checkpoints = [f for f in os.listdir(path) if f.endswith('.ckpt')]
    checkpoints.sort()
    checkpoint = checkpoints[-1]
    print("Using checkpoint: ", checkpoint)
    mlp_model = MLPModel.load_from_checkpoint(path + checkpoint, map_location=torch.device('mps'))

    trainer = Trainer(logger=False)
    predictions = trainer.predict(mlp_model, val_loader)
    predictions = torch.cat(predictions, dim=0)

    if dataset == "val":
        targets = torch.from_numpy(val_data['rating'].values)
        rmse = MeanSquaredError()
        mlp_rmse = rmse(predictions, targets).item()
        print(mlp_rmse)

    val_data['Prediction'] = predictions
    val_data['Prediction'].to_csv(f"lightning_logs/{version}/predictions_{dataset}.csv")

