import argparse
import datetime
import os
import random
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from ..metrics.output_losses import output_loss
from ..metrics.pca_plot_genomes import pca_plot
from . import functions as func


def train(args: argparse.ArgumentParser, generated_df: pd.DataFrame=None,
          out_dir: str=None) -> Tuple[nn.Module, nn.Module]:
    """
    Create the model.

    Args:
        args (argparse.ArgumentParser): arguments
        generated_df (pd.DataFrame, optional):\
            Specify the training data from sources other than
            the arguments. Defaults to None.
        out_dir (str, optional):\
            Specify the output directory from sources other than
            the arguments. Defaults to None.

    Returns:
        Tuple[nn.Module, nn.Module]: Generator, Discriminator
    """

    print('start training GAN.')

    # ----Set seed for reproducibility----
    manualSeed = 42
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Random Seed: ", manualSeed)


    if generated_df is None:
        inpt = args.train_file
        out_dir = args.out_dir
    else:
        inpt = generated_df
        out_dir = out_dir

    beta1 = args.beta1
    beta2 = args.beta2
    g_learn = args.g_learn
    d_learn = args.d_learn
    weight_decay = args.weight_decay
    epochs = args.epochs
    latent_size = args.latent_size
    alph = args.alph
    batch_size = args.batch_size
    gpu = args.gpu
    save_that = args.save_that
    label_noise = args.label_noise
    dropout = args.dropout
    norm = args.norm
    ag_size = args.ag_size

    delta = args.delta
    sigma = args.sigma
    max_per_sample_grad_norm = args.clip_value
    eps = args.eps

    # create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(f'{out_dir} already exists.')

    if epochs % 10 == 0:
        epochs += 1

    # write command line arguments to settings.txt
    with open(os.path.join(out_dir, 'settings_training.txt'), 'w') as f:
        for arg in vars(args):
            f.write(arg + ': ' + str(getattr(args, arg)) + '\n')
        f.write(f'datetime: {datetime.datetime.now()}')

    # device settings
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0)
                            else "cpu")  # set device to cpu or gpu

    df, df_noname, dataloader = func.load_data(
        inpt, batch_size, device, noise=args.label_noise)

    ag_size = len(df_noname)

    netG, netD = func.create_models(
        latent_size, df_noname.shape[1], gpu, device, alph, norm, dropout
    )

    # Optimizers for generator and critic
    d_optimizer = torch.optim.Adam(netD.parameters(), lr=d_learn,
                                betas=(beta1,beta2), weight_decay=weight_decay
                                )
    g_optimizer = torch.optim.Adam(netG.parameters(), lr=g_learn,
                                betas=(beta1,beta2), weight_decay=weight_decay
                                )

    # opacus
    if args.apply_dp:
        netD, d_optimizer, dataloader, privacy_engine\
            = func.use_opacus(netD=netD, d_optimizer=d_optimizer,
                              dataloader=dataloader, epsilon=eps,
                              delta=delta, epochs=epochs, sigma=sigma,
                              clip_value=max_per_sample_grad_norm,
                              use_poisson_sampling=args.use_poisson_sampling)

    # if use trained model
    if args.relearn:
        print(f'use trained model from <{args.open_model}>')
        func.load_models(
            netG=netG, netD=netD, g_opt=g_optimizer, d_opt=d_optimizer,
            path=os.path.join(args.open_model, f'{args.open_epochs}.pt'))

    y_real, y_fake = torch.ones([batch_size, 1]), torch.zeros([batch_size, 1])
    y_real = y_real.to(device)
    y_fake = y_fake.to(device)
    LABEL_REAL = 1.0
    LABEL_FAKE = 0.0
    losses = []
    criterion = torch.nn.BCELoss()

    # Training Loop
    print("Starting Training Loop...")
    print("device: ", device)
    start_time = time.time()
    for e in tqdm(range(epochs)):
        b = 0
        while b < len(dataloader):
            b += 1
            ###############################
            # (1) discriminator update
            #     maximize log(D(x)) + log(1 - D(G(z)))
            ###############################
            netD.zero_grad(set_to_none=True)

            # training on real data
            X_batch_real = next(iter(dataloader)).to(device)
            batch_size = X_batch_real.shape[0]
            y_real = torch.full([batch_size, 1], LABEL_REAL, device=device)
            d_pred_real = netD(X_batch_real)
            if label_noise != 0:
                noise = torch.Tensor(y_real.shape[0], y_real.shape[1])\
                    .uniform_(0, args.label_noise).to(device)
                errD_real = criterion(d_pred_real, y_real-noise)
            else:
                errD_real = criterion(d_pred_real, y_real)
            errD_real.backward()

            # training on fake data
            latent_samples = torch.randn((batch_size, latent_size)).to(device)
            X_batch_fake = netG(latent_samples)
            y_fake = torch.full([batch_size, 1], LABEL_FAKE, device=device)
            d_pred_fake = netD(X_batch_fake.detach())
            errD_fake = criterion(d_pred_fake, y_fake)
            errD_fake.backward()
            errD = errD_real + errD_fake
            d_optimizer.step()
            d_optimizer.zero_grad(set_to_none=True)

            ##############################
            # generator update
            # maximize log(D(G(z)))
            ##############################
            netG.zero_grad()
            d_pred = netD(X_batch_fake)
            errG = criterion(d_pred, y_real)
            errG.backward()
            g_optimizer.step()

        if args.apply_dp and sigma > 0:
            epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
            losses.append((errD.detach().to('cpu').numpy(),
                           errG.detach().to('cpu').numpy(), epsilon, delta))
            print(f"Epoch: {e+1}/{epochs} Discriminator loss: {errD:6.4f} \
                Generator loss: {errG:6.4f} ε: {epsilon:.5f} δ: {delta:.5f}")

        else:
            losses.append((errD.detach().to('cpu').numpy(),
                           errG.detach().to('cpu').numpy()))
            print(f"Epoch: {e+1}/{epochs} Discriminator loss: {errD:6.4f} \
                Generator loss: {errG:6.4f}")


        # save progress
        if e%save_that == 0 or e == epochs:
            if e == 0:
                continue

            if args.relearn:
                e = e + args.open_epochs

            #Save models
            func.save_models(netG=netG, netD=netD, g_opt=g_optimizer,
                    d_opt=d_optimizer, epo=e, path=out_dir)

            netG.eval()
            #Create AGs
            generated_genomes_df\
                = func.create_artificial_genomes(ag_size=ag_size,
                                                 latent_size=latent_size,
                                                 device=device, netG=netG,
                                                 out_dir=out_dir,
                                                 epoch=e, is_regen=False)

            #Output losses
            output_loss(losses=losses, e=e, out_dir=out_dir)

            #Make PCA
            pca_plot(df, generated_genomes_df, e, out_dir)
            netG.train()

    print("--- %s seconds ---" % (time.time() - start_time))

    return netG, netD
