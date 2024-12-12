import os
import argparse

from modules.constants import LocalPaths
from modules.constants import training
from pipeline import PrivacyProtectedArtificialGenomes



def define_task(parser: argparse.ArgumentParser):
    """Specify which tasks to perform with this execution.

    Args:
        parser (argparse.ArgumentParser): raw parser

    Returns:
        argparse.ArgumentParser: task_specified parser
    """
    parser.add_argument('--prepare', action='store_true',
                        help=f'Data Preparation')
    parser.add_argument('--train', action='store_true',
                        help=f"Model's Training")
    parser.add_argument('--regenerate', action='store_true',
                        help=f'Regenerate Artificial Genomes')
    parser.add_argument('--imputation', action='store_true',
                        help=f'Imputation')
    parser.add_argument('--wb_attack', action='store_true',
                        help=f'Membership Inference Attack\
                            (white-box settings)')
    return parser


def define_directory(parser: argparse.ArgumentParser):
    """Specify the directories

    Args:
        parser (argparse.ArgumentParser): raw parser

    Returns:
        argparse.ArgumentParser: task_specified parser
    """
    parser.add_argument('--work_dir', default=LocalPaths.WORK_DIR, required=True,
                        help=f'Work directory (default: {LocalPaths.WORK_DIR})')
    parser.add_argument('--data_dir', default='data',
                        help=f'Data directory under the "work_dir" (default: data)')
    return parser


def define_training(parser: argparse.ArgumentParser):
    """Define settings for training

    Args:
        parser (argparse.ArgumentParser): raw parser

    Returns:
        argparse.ArgumentParser: task_specified parser
    """
    # training data
    parser.add_argument('--train_file', type=str, default='training_data.hapt',
                        help=f'File name of the training data (under the \
                            data directory) (default: training_data.hapt)')
    # output directory
    parser.add_argument('--out_dir', type=str, default='baseline',
                        help=f'The directory to save the trained model\
                            and related files. (under the work_dir) (default: baseline)')

    # <<<<< Training Settings >>>>>
    parser.add_argument('--beta1', type=float, default=training.BETA1,
                        help=f'Beta1 value (default: {training.BETA1})')
    parser.add_argument('--beta2', type=float, default=training.BETA2,
                        help=f'Beta2 value (default: {training.BETA2})')
    parser.add_argument('--g_learn', type=float, default=training.G_LEARN,
                        help=f'g_learn value (default: {training.G_LEARN})')
    parser.add_argument('--d_learn', type=float, default=training.D_LEARN,
                        help=f'd_learn value (default: {training.D_LEARN})')
    parser.add_argument('--weight_decay', type=float, default=training.WEIGHT_DECAY,
                        help=f'weight_decay value (default: {training.WEIGHT_DECAY})')
    parser.add_argument('--epochs', '-e', type=int, default=training.EPOCHS,
                        help=f'epochs, don\'t forget +1 (default: {training.EPOCHS})')
    parser.add_argument('--latent_size', type=int, default=training.LATENT_SIZE,
                        help=f'latent_size (default: {training.LATENT_SIZE})')
    parser.add_argument('--alph', type=float, default=training.ALPH,
                        help=f'alpha value for LeakyReLU (default: {training.ALPH})')
    parser.add_argument('--batch_size', type=int, default=training.BATCH_SIZE,
                        help=f'batch_size (default: {training.BATCH_SIZE})')
    parser.add_argument('--gpu', type=int, default=training.GPU,
                        help=f'number of GPUs to use (default: {training.GPU})')
    parser.add_argument('--save_that', type=int, default=training.SAVE_THAT,
                        help=f'epoch interval for saving outputs (default: {training.SAVE_THAT})')
    parser.add_argument('--label_noise', type=float, default=training.LABEL_NOISE,
                        help=f'size of noise added to data/labels (default: {training.LABEL_NOISE})')
    parser.add_argument('--dropout', type=float, default=training.DROPOUT,
                        help=f'dropout rate (default: {training.DROPOUT})')
    parser.add_argument('--norm', default=training.NORM, choices=['None', 'Batch', 'Instance', 'Group'],
                        help=f'Normalization method (default: {training.NORM})')
    parser.add_argument('--ag_size', type=int, default=training.AG_SIZE,
                        help=f'Size of generated artificial genomes (default: {training.AG_SIZE})')

    # <<<<< Re-training Settings >>>>>
    parser.add_argument('--relearn', action='store_true',
                        help='Set when performing additional training')
    parser.add_argument('--open_model', default='', type=str,
                        help='Set the directory name of the base model when performing additional training')
    parser.add_argument('--open_epochs', default=0, type=int,
                        help='Specify the number of epochs of the base model when performing additional training')

    # <<<<< Differential Privacy Settings >>>>>
    parser.add_argument('--apply_dp', action='store_true',
                        help='Set when using differential privacy')
    parser.add_argument('--delta', type=float, default=training.DELTA,
                        help=f'for differential privacy (default: {training.DELTA})')
    parser.add_argument('--sigma', type=float, default=training.SIGMA,
                        help=f'for noise_multiplier (default: {training.SIGMA})')
    parser.add_argument('--clip_value', '-c', type=float, default=training.MAX_PER_SAMPLE_GRAD_NORM,
                        help=f'clipping value in dp-sgd (default: {training.MAX_PER_SAMPLE_GRAD_NORM})')
    parser.add_argument('--eps', type=float, default=training.EPS,
                        help=f'target epsilon (default: {training.EPS})')
    parser.add_argument('--use_poisson_sampling', action='store_true',
                        help='Set when using poisson_sampling with opacus')

    return parser


def define_evaluation(parser: argparse.ArgumentParser):
    """define settings for evaluation

    Args:
        parser (argparse.ArgumentParser): raw parser

    Returns:
        argparse.ArgumentParser: task_specified parser
    """
    # <<<<< target model of the evaluation >>>>>
    parser.add_argument('--model_dir', type=str, default='models',
                        help=f'Model directory under the work_dir\
                            (default: models)')
    parser.add_argument('--model_name', type=str, default='baseline.pt',
                        help=f'Model file name under the model_dir\
                            (default: baseline.pt)')
    parser.add_argument('--model_type', default='Baseline', type=str,
                        choices=['Baseline', 'DP', 'Clipping'],
                        help=f'Model type for membership inference\
                            (default: Baseline)')

    # <<<<< genotype imputation >>>>>
    parser.add_argument('--imputation_target_size',
                        default=508, type=int,
                        help=f'Number of haplotypes for imputation target\
                            (=test data size) (default: 508)')
    parser.add_argument('--ref_type', default='1KG', type=str,
                        choices=['1KG', 'GAN', 'DP', 'Clip'],
                        help=f'Reference type for genotype imputation\
                            (default: 1KG)')
    parser.add_argument('--ref_haps_size', default=4000, type=int,
                        help=f'Number of haplotypes for the reference of \
                            genotype imputation. (default: 4000)')
    parser.add_argument('--model_epochs', type=int, default=16000,
                        help=f"Which epoch's output to use as reference.\
                            (default: 16000)")
    parser.add_argument('--ag_file_name', type=str,
                        default='16000_output.hapt',
                        help=f'The file name of AG to be used as reference\
                        (under model_dir) (default: 16000_output.hapt)')


    # <<<<< membership inference attack >>>>>
    parser.add_argument('--test_file', type=str, default='test_data.hapt',
                        help=f'File name of the test data (under the \
                            data directory) (default: test_data.hapt)')

    return parser


if __name__ == "__main__":
    # prepare parser
    parser = argparse.ArgumentParser()

    # specify task
    parser = define_task(parser)

    # specify directories
    parser = define_directory(parser)

    # specify information about model training
    parser = define_training(parser)

    # specify information about evaluation
    parser = define_evaluation(parser)

    args = parser.parse_args()

    # Modify the paths according to the directory specified in work_dir.
    path = LocalPaths(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    args.data_dir = os.path.join(path.WORK_DIR, args.data_dir)
    args.model_dir = os.path.join(path.WORK_DIR, args.model_dir)
    args.model_name = os.path.join(args.model_dir, args.model_name)
    args.out_dir = os.path.join(path.WORK_DIR, args.out_dir)
    args.train_file = os.path.join(path.TRAINING_DATA_DIR, args.train_file)
    args.test_file = os.path.join(path.TRAINING_DATA_DIR, args.test_file)

    pipeline = PrivacyProtectedArtificialGenomes(args)
    pipeline.main()
