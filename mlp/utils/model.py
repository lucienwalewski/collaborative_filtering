import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError

class MLPModel(pl.LightningModule):
    def __init__(self, args, user_num, movie_num):
        super().__init__()
        self.save_hyperparameters()

        self.user_num = user_num
        self.movie_num = movie_num
        self.factor_num = args.factor_num
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.user_embedding = nn.Embedding(user_num, int(self.factor_num * (2 ** (self.num_layers - 1))))
        self.movie_embedding = nn.Embedding(movie_num, int(self.factor_num * (2 ** (self.num_layers - 1))))

        mlp_modules = []
        for i in range(self.num_layers):
            input_size = self.factor_num * (2 ** (self.num_layers - i))
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

        self.out_dim = args.out_dim  # can be 1 or 5
        self.predict_layer = nn.Linear(self.factor_num,  self.out_dim)
        self.loss_fn = nn.CrossEntropyLoss() if self.out_dim != 1 else nn.MSELoss()

        self.rmse = MeanSquaredError()

    def forward(self, batch):

        user, movie, rating = batch

        user_embedding = self.user_embedding(user)
        movie_embedding = self.movie_embedding(movie)

        mlp_input = torch.cat([user_embedding, movie_embedding], dim=1)
        mlp_output = self.mlp_layers(mlp_input)
        logits = self.predict_layer(mlp_output)

        if self.out_dim == 1:
            rating = rating.float().view(-1, 1)
        loss = self.loss_fn(logits, rating)

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, logits = self.forward(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.forward(batch)
        user, movie, rating = batch

        if self.out_dim != 1:
            probs = torch.softmax(logits, dim=1)
            # preds = torch.argmax(probs, dim=1) + 1
            preds = torch.sum(probs * torch.arange(1, 6, device=self.device), dim=1)
        else:
            preds = logits.squeeze() + 1

        rmse = self.rmse(preds, rating + 1)
        metrics = {
            "val_loss": loss,
            "val_rmse": rmse,
        }
        self.log_dict(metrics)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        loss, logits = self.forward(batch)
        if self.out_dim != 1:
            probs = torch.softmax(logits, dim=1)
            # preds = torch.argmax(probs, dim=1) + 1
            preds = torch.sum(probs * torch.arange(1, 6, device=self.device), dim=1)
        else:
            preds = logits.squeeze() + 1
        return preds

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)