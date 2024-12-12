
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ...constants.local_paths import LocalPaths
from .models_GAN import (Discriminator, DiscriminatorWithDropout,
                         Generator, GeneratorWithDropout)


def save_models(netG: nn.Module, netD: nn.Module,
                g_opt: Optimizer, d_opt: Optimizer,
                epo: int, path: str):
    """
    save the created model and optimizer

    Args:
        netG (nn.Module): Generator
        netD (nn.Module): Discriminator
        g_opt (Optimizer): Optimizer of the generator
        d_opt (Optimizer): Optimizer of the discriminator
        epo (int): Epochs
        path (str): Path
    """
    torch.save(
        {
            'Generator': netG.state_dict(),
            'Discriminator': netD.state_dict(),
            'G_optimizer': g_opt.state_dict(),
            'D_optimizer': d_opt.state_dict()
        },
        os.path.join(path, f'{str(epo)}.pt')
    )


def load_models(netG: nn.Module, netD: nn.Module,
                g_opt: Optimizer, d_opt: Optimizer, path: str):
    """
    load models and optimizers from given path

    Args:
        netG (nn.Module): Generator
        netD (nn.Module): Discriminator
        g_opt (Optimizer): Optimizer of the generator
        d_opt (Optimizer): Optimizer of the discriminator
        path (str): Path
    """
    checkpoint = torch.load(path)
    netG.load_state_dict(checkpoint['Generator'])
    netD.load_state_dict(checkpoint['Discriminator'])
    g_opt.load_state_dict(checkpoint['G_optimizer'])
    d_opt.load_state_dict(checkpoint['D_optimizer'])


def create_models(
    latent_size: int, data_shape: int, gpu: int,
    device: torch.device, alph: float, norm: str,
    dropout: float) -> Tuple[nn.Module, nn.Module]:

    """
    Create GAN from given parameters

    Args:
        latent_size (int): Size of the latent vector
        data_shape (int): Shape of the data
        gpu (int): Avairable gpu
        device (torch.device): Device
        alph (float): For LeakyReLU
        norm (str): Regularization method
        dropout (float): Dropout rate

    Returns:
        Tuple[nn.Module, nn.Module]: Generator, Discriminator
    """
    # Create the generator without dropout
    if norm == 'None' and dropout == 0:
        # Create the generator
        netG = Generator(latent_size=latent_size,
                        data_shape=data_shape,
                        gpu=gpu, device=device,
                        alph=alph).to(device)
        netG = netG.float()
        if (device.type == 'cuda') and (gpu > 1):
            netG = nn.DataParallel(netG, list(range(gpu)))
        netG.to(device)

        # Create the discriminator
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
        # Create the generator
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

        # Create the discriminator
        netD = DiscriminatorWithDropout(data_shape=data_shape,
                            latent_size=latent_size, gpu=gpu,
                            device=device, alph=alph,
                            norm=norm, dropout=dropout).to(device)
        netD = netD.float()
        if (device.type == 'cuda') and (gpu > 1):
            netD = nn.DataParallel(netD, list(range(gpu)))
        netD.to(device)
    return netG, netD


# Prepare the training data
def load_data(inpt: str, batch_size: int, device: torch.device, noise: float)\
    -> Tuple[pd.DataFrame, pd.DataFrame, DataLoader]:
    """
    Prepare dataset from hapt file

    Args:
        inpt (str): Path to the original data
        batch_size (int): Batch size
        device (torch.device): Device
        noise (float): The amount of the noise added to the data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, DataLoader]: Dataset
    """

    df = pd.read_csv(inpt, sep=' ', header=None)
    df = df.sample(frac=1).reset_index(drop=True)
    df_noname = df.drop(df.columns[0:2], axis=1)
    df_noname = df_noname.values
    if noise > 0:
        df_noname = df_noname - \
            np.random.uniform(0, noise, size=(
                df_noname.shape[0], df_noname.shape[1]))
    df_noname = torch.Tensor(df_noname)
    df_noname = df_noname.to(device)
    dataloader = DataLoader(
        df_noname, batch_size=batch_size, shuffle=True, pin_memory=False,
    )
    return df, df_noname, dataloader


# set opacus
def use_opacus(
    netD: nn.Module,
    d_optimizer: Optimizer,
    dataloader: DataLoader,
    epsilon: float,
    delta: float,
    epochs: int,
    sigma: float,
    clip_value:float,
    use_poisson_sampling: bool
) -> Tuple[nn.Module, Optimizer, DataLoader, PrivacyEngine]:
    """
    Use opacus (gradient clipping, differential privacy).

    Args:
        netD (nn.Module): Discriminator
        d_optimizer (Optimizer): Optimizer of the discriminator
        dataloader (DataLoader): Dataloader
        epsilon (float): Target epsilon of the differential privacy
        delta (float): Delta value of the differential privacy
        epochs (int): Training epochs
        sigma (float): Noise_multiplier
        clip_value (float): Decides gradient clipping
        use_poisson_sampling (bool): Apply poisson sampling

    Returns:
        Tuple[nn.Module, Optimizer, DataLoader, PrivacyEngine]:
        Elements with opacus applied
    """

    # Apply opacus
    privacy_engine = PrivacyEngine()
    if sigma == -1:
        netD, d_optimizer, dataloader\
            = privacy_engine.make_private_with_epsilon(
                module=netD,
                optimizer=d_optimizer,
                data_loader=dataloader,
                target_epsilon=epsilon,
                target_delta=delta,
                epochs=epochs,
                max_grad_norm=clip_value,
                )
        print(f"Using sigma={d_optimizer.noise_multiplier}\
            and C={clip_value}")
    else:
        netD, d_optimizer, dataloader = privacy_engine.make_private(
            module=netD,
            optimizer=d_optimizer,
            data_loader=dataloader,
            noise_multiplier=sigma,
            max_grad_norm=clip_value,
            poisson_sampling=use_poisson_sampling
        )
        print(f"Using sigma={d_optimizer.noise_multiplier}\
            and C={clip_value}")
    return netD, d_optimizer, dataloader, privacy_engine


def create_artificial_genomes(
    ag_size: int, latent_size: int,
    device: torch.device, netG: nn.Module,
    out_dir: str, epoch: int, is_regen: bool,
    filepath: str = '') -> pd.DataFrame:
    """ generate artificial genomes

    Args:
        ag_size (int): The size of generated data
        latent_size (int): Noise dimention
        device (torch.device): Device information (GPU or CPU)
        netG (nn.Module): Generator
        out_dir (str): Output directory
        epoch (int): Epochs
        is_regen (bool): Set True if generate from trained model
        filepath (str): Filepath for regenerate

    Returns:
        pd.DataFrame: Generated artificial genomes
    """
    latent_samples = torch.randn((ag_size, latent_size)).to(device) #noise
    generated_genomes = netG(latent_samples)
    generated_genomes = generated_genomes.to('cpu')
    generated_genomes[generated_genomes < 0] = 0
    generated_genomes = generated_genomes.round()
    generated_genomes_df = pd.DataFrame(generated_genomes.detach().numpy())
    generated_genomes_df = generated_genomes_df.astype(int)
    gen_names = list()
    for i in range(0,len(generated_genomes_df)):
        gen_names.append('AG'+str(i))
    generated_genomes_df.insert(loc=0, column='Type', value="AG")
    generated_genomes_df.insert(loc=1, column='ID', value=gen_names)
    generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))

    #Output AGs in hapt format
    if is_regen:
        num = 1
        filename = filepath + f'{num}'
        if os.path.exists(filename):
            while(1):
                num+=1
                filename = filepath + f'{num}'
                if not os.path.exists(filename):
                    break
    else:
        filename = os.path.join(out_dir, str(epoch)+'_output.hapt')
    generated_genomes_df.to_csv(
        filename, sep=" ", header=False, index=False)
    return generated_genomes_df


def regenerate_artificial_genomes(args):
    """
    Generate artificial genomes from specified model.

    Args:
        args: Parsed command-line arguments
    """
    # Path setting
    path = LocalPaths(args.work_dir)
    model_dir = args.model_dir

    # load settings used during training
    with open(os.path.join(model_dir, 'settings_training.txt'), 'r') as f:
        settings = f.readlines()
        f.close()

    ## delete '\n' in each line
    settings = [i.strip('\n') for i in settings]

    ## convert string to appropreate types such as bool or integer
    s = {}
    for i in settings:
        key, value = i.split(': ')
        if value == 'True':
            value = True
        elif value == 'False':
            value = False
        else:
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
        s[key] = value

    # load models
    ## preparation
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and s['gpu'] > 0) else "cpu")
    _, df_noname, dataloader = load_data(
        path.TRAINING_DATA_PATH, s['batch_size'], device, s['label_noise'])

    netG, netD = create_models(
        s['latent_size'], df_noname.shape[1], s['gpu'], device, s['alph'],
        norm=s['norm'], dropout=s['dropout'],
    )

    # Optimizers for generator and critic
    d_optimizer = torch.optim.Adam(netD.parameters(), lr=s['d_learn'],
                            betas=(s['beta1'],s['beta2']), weight_decay=s['weight_decay']
                            )
    g_optimizer = torch.optim.Adam(netG.parameters(), lr=s['g_learn'],
                            betas=(s['beta1'],s['beta2']), weight_decay=s['weight_decay']
                            )
    if s['apply_dp']:
        netD, d_optimizer, dataloader, _\
            = use_opacus(netD=netD, d_optimizer=d_optimizer,
                         dataloader=dataloader, epsilon=s['eps'],
                         delta=s['delta'], epochs=s['epochs'],
                         sigma=s['sigma'], clip_value=s['clip_value'],
                         use_poisson_sampling=s['use_poisson_sampling'])

    load_models(netG=netG, netD=netD, g_opt=g_optimizer, d_opt=d_optimizer,
                path=args.model_name)

    print(f'\n From {args.model_name}, generate {args.ag_size} haplotypes.\n')

    # define generated file name for regenerate
    filepath = args.model_name.split('.pt')[0] + '_output_regen.hapt'
    _ = create_artificial_genomes(
        ag_size=args.ag_size, latent_size=s['latent_size'],
        device=device, netG=netG, out_dir=model_dir,
        epoch=args.model_epochs, is_regen=True,
        filepath=filepath)


