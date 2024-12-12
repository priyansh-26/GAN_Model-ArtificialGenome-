# default parameters for training
## learning rate
G_LEARN = 0.0001
D_LEARN = 0.0008

## epochs
EPOCHS = 10001

## latent vector size
LATENT_SIZE = 600

## LeakyReLU's parameter
ALPH = 0.01

## batch_size
BATCH_SIZE = 32

## number of GPU
GPU = 1

## how often to save (in epochs)
SAVE_THAT = 500

## amount of noise to add
LABEL_NOISE = 0.1

## dropout rate
DROPOUT = 0

## regularization method
NORM = 'None'

## size of generated artificial genomes
AG_SIZE = 4000

## for adam
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.0001

## for opacus
DELTA = 1e-5
SIGMA = -1
MAX_PER_SAMPLE_GRAD_NORM = 1.0
EPS = 1.0
