import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def output_loss(losses: list, e, out_dir):
    pd.DataFrame(losses).to_csv(
        os.path.join(out_dir, str(e)+'_losses.txt'),
        sep=" ", header=False, index=False)
    fig, ax = plt.subplots()
    plt.plot(np.array([losses]).T[0], label='Discriminator')
    plt.plot(np.array([losses]).T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    fig.savefig(os.path.join(out_dir, str(e)+'_loss.pdf'), format='pdf')
    plt.close()
