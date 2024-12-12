from typing import Tuple

import torch
import torch.nn as nn

from .models_GAN import (Discriminator, DiscriminatorWithDropout,
                         Generator, GeneratorWithDropout)


def create_models(
    latent_size: int, data_shape: int, gpu: int, device: torch.device,
    alph: float, norm: str, dropout: float) -> Tuple[nn.Module, nn.Module]:

    if norm == 'None' and dropout == 0:
        # Create the GAN without dropout
        ## Create the generator
        netG = Generator(latent_size=latent_size,
                        data_shape=data_shape,
                        gpu=gpu, device=device,
                        alph=alph).to(device)
        netG = netG.float()
        if (device.type == 'cuda') and (gpu > 1):
            netG = nn.DataParallel(netG, list(range(gpu)))
        netG.to(device)

        ## Create the discriminator
        netD = Discriminator(
            data_shape=data_shape,
            latent_size=latent_size, gpu=gpu,
            device=device, alph=alph
            ).to(device)
        netD = netD.float()
        if (device.type == 'cuda') and (gpu > 1):
            netD = nn.DataParallel(netD, list(range(gpu)))
        netD.to(device)
    else:
        # Create the GAN with dropout
        ## Create the generator
        netG = GeneratorWithDropout(
            latent_size=latent_size,
            data_shape=data_shape,
            gpu=gpu, device=device,
            alph=alph,
            norm=norm, dropout=dropout
            ).to(device)
        netG = netG.float()
        if (device.type == 'cuda') and (gpu > 1):
            netG = nn.DataParallel(netG, list(range(gpu)))
        netG.to(device)

        ## Create the discriminator
        netD = DiscriminatorWithDropout(
            data_shape=data_shape,
            latent_size=latent_size, gpu=gpu,
            device=device, alph=alph,
            norm=norm, dropout=dropout
            ).to(device)
        netD = netD.float()
        if (device.type == 'cuda') and (gpu > 1):
            netD = nn.DataParallel(netD, list(range(gpu)))
        netD.to(device)
    return netG, netD
