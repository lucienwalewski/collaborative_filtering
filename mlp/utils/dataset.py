import torch
from torch.utils.data import Dataset

class MLPDataset(Dataset):
    def __init__(self, data, user_num, movie_num):

        self.data = data
        self.user_num = user_num
        self.movie_num = movie_num

        '''
        self.mean = mean
        self.std = std

        self.normalize_by = "movie"
        if mean is None or std is None:
            self.mean = self.data.groupby(by=self.normalize_by)['rating'].mean()
            self.std = self.data.groupby(by=self.normalize_by)['rating'].std()

        self.data.rating = (self.data.rating - self.mean.values[self.data[self.normalize_by]]) / self.std.values[
            self.data[self.normalize_by]]
        '''

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, movie, rating = torch.tensor(self.data.iloc[idx]).type(torch.long)
        return user, movie, rating