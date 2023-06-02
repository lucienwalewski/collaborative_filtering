import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
import os
import numpy as np

class MLPModel(pl.LightningModule):
    def __init__(self, args, user_num, movie_num, mean=None, std=None):
        super().__init__()
        self.save_hyperparameters()

        # define parameters
        self.model = args.model # "mlp" or "mf" or "ncf"
        self.user_num = user_num
        self.movie_num = movie_num
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.mean = mean
        self.std = std
        self.normalize_by = args.normalize_by

        # define mlp part
        if self.model == "mlp" or self.model == "ncf":
            self.num_layers = args.num_layers
            self.dropout = args.dropout
            self.mlp_out_dim = args.mlp_out_dim

            self.mlp_embedding_dim = int(self.mlp_out_dim * (2 ** (self.num_layers - 1)))
            self.mlp_user_embedding = nn.Embedding(self.user_num, self.mlp_embedding_dim)
            self.mlp_movie_embedding = nn.Embedding(self.movie_num, self.mlp_embedding_dim)

            mlp_modules = []
            for i in range(self.num_layers):
                input_size = self.mlp_out_dim * (2 ** (self.num_layers - i))
                mlp_modules.append(nn.Dropout(p=self.dropout))
                mlp_modules.append(nn.Linear(input_size, input_size // 2))
                mlp_modules.append(nn.ReLU())
            self.mlp_layers = nn.Sequential(*mlp_modules)

        # define mf part
        if self.model == "mf" or self.model == "ncf":
            self.mf_embedding_dim = args.mf_embedding_dim
            self.mf_user_embedding = nn.Embedding(self.user_num, self.mf_embedding_dim)
            self.mf_movie_embedding = nn.Embedding(self.movie_num, self.mf_embedding_dim)

        # define predict layer
        if self.model == "mlp":
            input_dim = self.mlp_out_dim
        elif self.model == "mf":
            input_dim = self.mf_embedding_dim
        else:
            input_dim = self.mf_embedding_dim + self.mlp_out_dim

        self.out_dim = args.out_dim  # can be 1 or 5
        self.predict_layer = nn.Linear(input_dim,  self.out_dim)
        self.loss_fn = nn.CrossEntropyLoss() if self.out_dim != 1 else nn.MSELoss()

        # define metrics
        self.rmse = MeanSquaredError()

        # init with pretrained
        if args.mf_pretrained != "":
            if args.mf_pretrained == "ALS":
                mf_user_embedding = np.load("lightning_logs/ALS/P.npy")
                mf_movie_embedding = np.load("lightning_logs/ALS/Q.npy")
                self.mf_user_embedding.weight.data = torch.tensor(mf_user_embedding.T).float()
                self.mf_movie_embedding.weight.data = torch.tensor(mf_movie_embedding.T).float()
                self.predict_layer.weight.data = torch.ones_like(self.predict_layer.weight.data)
                self.predict_layer.bias.data = torch.zeros_like(self.predict_layer.bias.data)
            else:
                mf_model_path = f"lightning_logs/{args.mf_pretrained}/checkpoints/"
                checkpoints = [f for f in os.listdir(mf_model_path) if f.endswith('.ckpt')]
                step_nr = [int(f.split(".")[0].split("=")[-1]) for f in checkpoints]
                checkpoint = checkpoints[np.argmax(step_nr)]
                print(args.mf_pretrained + checkpoint)
                mf_model = torch.load(mf_model_path + checkpoint, map_location=self.device)
                self.mf_user_embedding.weight.data = mf_model['state_dict']['mf_user_embedding.weight']
                self.mf_movie_embedding.weight.data = mf_model['state_dict']['mf_movie_embedding.weight']
                self.predict_layer.weight.data[:,:self.mf_embedding_dim] = mf_model['state_dict']['predict_layer.weight']
                self.predict_layer.bias.data[:self.mf_embedding_dim] = mf_model['state_dict']['predict_layer.bias']

        if args.mlp_pretrained != "":
            mlp_model_path = f"lightning_logs/{args.mlp_pretrained}/checkpoints/"
            checkpoints = [f for f in os.listdir(mlp_model_path) if f.endswith('.ckpt')]
            step_nr = [int(f.split(".")[0].split("=")[-1]) for f in checkpoints]
            checkpoint = checkpoints[np.argmax(step_nr)]
            print(args.mlp_pretrained + checkpoint)
            mlp_model = torch.load(mlp_model_path + checkpoint, map_location=self.device)
            self.mlp_user_embedding.weight.data = mlp_model['state_dict']['mlp_user_embedding.weight']
            self.mlp_movie_embedding.weight.data = mlp_model['state_dict']['mlp_movie_embedding.weight']
            self.predict_layer.weight.data[:,-self.mlp_out_dim:] = mlp_model['state_dict']['predict_layer.weight']
            self.predict_layer.bias.data[-self.mlp_out_dim:] = mlp_model['state_dict']['predict_layer.bias']

            for idx, layer in enumerate(self.mlp_layers):
                if isinstance(layer, nn.Linear):
                    layer.weight.data = mlp_model['state_dict'][f'mlp_layers.{idx}.weight']
                    layer.bias.data = mlp_model['state_dict'][f'mlp_layers.{idx}.bias']

        if args.mlp_pretrained != "" and args.mf_pretrained != "":
            self.predict_layer.weight.data = self.predict_layer.weight.data / 2
            self.predict_layer.bias.data = self.predict_layer.bias.data / 2

    def forward(self, batch):

        user, movie, rating = batch

        # send trough mlp
        if self.model == "mlp" or self.model == "ncf":
            mlp_user_embedding = self.mlp_user_embedding(user)
            mlp_movie_embedding = self.mlp_movie_embedding(movie)
            mlp_input = torch.cat([mlp_user_embedding, mlp_movie_embedding], dim=1)
            mlp_output = self.mlp_layers(mlp_input)

        # send trough mf
        if self.model == "mf" or self.model == "ncf":
            mf_user_embedding = self.mf_user_embedding(user)
            mf_movie_embedding = self.mf_movie_embedding(movie)
            mf_output = mf_user_embedding * mf_movie_embedding

        # send trough prediction layer
        if self.model == "mlp":
            concat = mlp_output
        elif self.model == "mf":
            concat = mf_output
        else:
            concat = torch.cat([mf_output, mlp_output], dim=1)

        logits = self.predict_layer(concat)

        rating = rating.float().view(-1, 1) if self.out_dim == 1 else rating.long() - 1
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
            preds = torch.sum(probs * torch.arange(1, 6, device=self.device), dim=1)
        else:
            if self.normalize_by != "":
                normalize_by = user if self.normalize_by == "user" else movie
                std = self.std.to(self.device)[normalize_by]
                mean = self.mean.to(self.device)[normalize_by]
                preds = logits.squeeze() * std + mean
                rating = rating * std + mean
            else:
                preds = logits.squeeze()

        rmse = self.rmse(preds, rating)
        metrics = {
            "val_loss": loss,
            "val_rmse": rmse,
        }
        self.log_dict(metrics)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        loss, logits = self.forward(batch)
        user, movie, rating = batch

        if self.out_dim != 1:
            probs = torch.softmax(logits, dim=1)
            preds = torch.sum(probs * torch.arange(1, 6, device=self.device), dim=1)
        else:
            if self.normalize_by != "":
                normalize_by = user if self.normalize_by == "user" else movie
                std = self.std.to(self.device)[normalize_by]
                mean = self.mean.to(self.device)[normalize_by]
                preds = logits.squeeze() * std + mean
            else:
                preds = logits.squeeze()

        return preds

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)