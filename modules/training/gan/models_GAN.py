import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_size: int, data_shape: int,
                 gpu: int, device: torch.device, alph: float):
        super(Generator, self).__init__()

        #parameters initialization
        self.latent_size = latent_size
        self.alph = alph
        self.gpu = gpu
        self.data_shape = data_shape
        self.device = device

        #Blocks
        self.block1 = nn.Sequential(
            nn.Linear(latent_size, int(data_shape//1.2)),
            nn.LeakyReLU(alph),
            nn.Linear(int(data_shape//1.2), int(data_shape//1.1)),
            nn.LeakyReLU(alph),
            nn.Linear(int(data_shape//1.1), data_shape),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.block1(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, latent_size: int, data_shape: int,
                 gpu: int, device: torch.device, alph: float):
        super(Discriminator, self).__init__()

        #parameters initialization
        self.latent_size = latent_size
        self.alph = alph
        self.gpu = gpu
        self.data_shape = data_shape
        self.device = device

        self.block1 = nn.Sequential(
            nn.Linear(data_shape, data_shape//2),
            nn.LeakyReLU(alph),
            nn.Linear(data_shape//2, data_shape//3),
            nn.LeakyReLU(alph),
            nn.Linear(data_shape//3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        return x

class GeneratorWithDropout(nn.Module):
    def __init__(self, latent_size: int, data_shape: int,
                 gpu: int, device: torch.device, alph: float,
                 norm: str, dropout: float):
        super(GeneratorWithDropout, self).__init__()

        #parameters initialization
        self.latent_size = latent_size
        self.alph = alph
        self.gpu = gpu
        self.data_shape = data_shape
        self.device = device
        self.norm = norm
        self.dropout = dropout

        # create a block based on the specified normalization layer
        if self.norm == 'Instance':
            self.block = nn.Sequential(
                nn.Linear(latent_size, int(data_shape//1.2)),
                nn.InstanceNorm1d(int(data_shape//1.2)),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(int(data_shape//1.2), int(data_shape//1.1)),
                nn.InstanceNorm1d(int(data_shape//1.1)),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(int(data_shape//1.1), data_shape),
                nn.Tanh()
            )
        elif self.norm == 'Batch':
            self.block = nn.Sequential(
                nn.Linear(latent_size, int(data_shape//1.2)),
                nn.BatchNorm1d(int(data_shape//1.2)),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(int(data_shape//1.2), int(data_shape//1.1)),
                nn.BatchNorm1d(int(data_shape//1.1)),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(int(data_shape//1.1), data_shape),
                nn.Tanh()
            )
        elif self.norm == 'Group':
            self.block = nn.Sequential(
                nn.Linear(latent_size, int(data_shape//1.2)),
                nn.GroupNorm(1, int(data_shape//1.2)),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(int(data_shape//1.2), int(data_shape//1.1)),
                nn.GroupNorm(1, int(data_shape//1.1)),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(int(data_shape//1.1), data_shape),
                nn.Tanh()
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(latent_size, int(data_shape//1.2)),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(int(data_shape//1.2), int(data_shape//1.1)),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(int(data_shape//1.1), data_shape),
                nn.Tanh()
            )

    def forward(self, x):
        x = self.block(x)
        return x

class DiscriminatorWithDropout(nn.Module):
    def __init__(self, latent_size: int, data_shape: int,
                 gpu: int, device: torch.device, alph: float,
                 norm: float, dropout: float):

        super(DiscriminatorWithDropout, self).__init__()

        #parameters initialization
        self.latent_size = latent_size
        self.alph = alph
        self.gpu = gpu
        self.data_shape = data_shape
        self.device = device
        self.norm = norm
        self.dropout = dropout

        # create a block based on the specified normalization layer
        if self.norm == 'Instance':
            self.block = nn.Sequential(
                nn.Linear(data_shape, data_shape//2),
                nn.InstanceNorm1d(data_shape//2),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(data_shape//2, data_shape//3),
                nn.InstanceNorm1d(data_shape//3),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(data_shape//3, 1),
                nn.Sigmoid()
            )
        elif self.norm == 'Batch':
            self.block = nn.Sequential(
                nn.Linear(data_shape, data_shape//2),
                nn.BatchNorm1d(data_shape//2),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(data_shape//2, data_shape//3),
                nn.BatchNorm1d(data_shape//3),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(data_shape//3, 1),
                nn.Sigmoid()
            )
        elif self.norm == 'Group':
            self.block = nn.Sequential(
                nn.Linear(data_shape, data_shape//2),
                nn.GroupNorm(1, data_shape//2),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(data_shape//2, data_shape//3),
                nn.GroupNorm(1, data_shape//3),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(data_shape//3, 1),
                nn.Sigmoid()
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(data_shape, data_shape//2),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(data_shape//2, data_shape//3),
                nn.LeakyReLU(alph),
                nn.Dropout(self.dropout),
                nn.Linear(data_shape//3, 1),
                nn.Sigmoid()
            )
    def forward(self, x):
        x = self.block(x)
        return x
