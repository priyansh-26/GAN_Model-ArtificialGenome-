import os

import numpy as np
import pandas as pd
# import opacus as pd
import torch
from opacus import PrivacyEngine
# import opacus 

from ..constants import LocalPaths
from ..training.gan.functions import create_models, load_models
from .mia_functions import mia
from .save_results import draw_accuracy_bar, save_mia_results_as_txt


class MembershipInferenceAttack:
    def __init__(self, args) -> None:
        self.__args = args
        self.__model_type = args.model_type

        self.__wb_attack = args.wb_attack

        self.__device = None
        self.__batch_size = args.batch_size
        self.__model_dir = args.model_dir
        self.__model_path = args.model_name

        self.__latent_size = args.latent_size
        self.__alph = args.alph
        self.__norm = args.norm
        self.__dropout = args.dropout
        self.__gpu = args.gpu
        self.__label_noise = args.label_noise

        self.__apply_dp = args.apply_dp
        self.__sigma = args.sigma
        self.__eps = args.eps
        self.__delta = args.delta
        self.__epochs = args.epochs
        self.__max_per_sample_grad_norm = args.clip_value
        self.__use_poisson_sampling = args.use_poisson_sampling

        self.__train_file = args.train_file
        self.__test_file = args.test_file

        self.__train_size = 0
        self.__test_size = 0

        self.__data_shape = 0

        self.__train_loader = None
        self.__train_loader_1 = None
        self.__test_loader_1 = None

        self.__netG = None
        self.__netD = None

        # Paths
        self.__path = LocalPaths(WORK_DIR=self.__args.work_dir)

        # check whether model_dir correctly specified
        if not os.path.exists(self.__model_dir):
            raise FileNotFoundError(f'{self.__model_dir} is not found.')

        # Set save destination
        self.__save_dir = os.path.join(
            self.__path.MEMINF_RESULTS_DIR,
            self.__model_dir.split('/')[-1]
        )
        os.makedirs(self.__save_dir, exist_ok=True)


    def _set_device(self):
        self.__device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and self.__gpu > 0)\
                else "cpu")


    def _load_data(self) -> None:
        df_train = pd.read_csv(
            self.__train_file,
            sep=' ', header=None
        )
        df_test = pd.read_csv(
            self.__test_file,
            sep=' ', header=None
        )

        self.__train_size = len(df_train)
        self.__test_size = len(df_test)

        # train_data operation
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_train_noname = df_train.drop(df_train.columns[0:2], axis=1)
        df_train_noname = df_train_noname.values
        df_train_noname = df_train_noname\
            - np.random.uniform(
                0, self.__label_noise,
                size=(df_train_noname.shape[0], df_train_noname.shape[1]))
        df_train_noname = torch.Tensor(df_train_noname)
        df_train_noname = df_train_noname.to(self.__device)
        self.__train_loader = torch.utils.data.DataLoader(
            df_train_noname, batch_size=self.__batch_size,
            shuffle=True, pin_memory=False
        )
        self.__train_loader_1 = torch.utils.data.DataLoader(
            df_train_noname, batch_size=self.__train_size,
            shuffle=True, pin_memory=False
        )

        # test_data operation
        df_test = df_test.sample(frac=1).reset_index(drop=True)
        df_test_noname = df_test.drop(df_test.columns[0:2], axis=1)
        df_test_noname = df_test_noname.values
        df_test_noname = df_test_noname\
            - np.random.uniform(
                0, self.__label_noise,
                size=(df_test_noname.shape[0], df_test_noname.shape[1]))
        df_test_noname = torch.Tensor(df_test_noname)
        df_test_noname = df_test_noname.to(self.__device)
        test_loader = torch.utils.data.DataLoader(
            df_test_noname, batch_size=self.__batch_size,
            shuffle=True, pin_memory=False
        )
        self.__test_loader_1 = torch.utils.data.DataLoader(
            df_test_noname, batch_size=self.__test_size,
            shuffle=True, pin_memory=False
        )
        self.__data_shape = df_train_noname.shape[1]


    def _print_baseline(self):
        """
        Output the target model and baseline for membership inference
        """
        print('')
        print(f'--> MODEL FROM {self.__model_path}')
        print(f'--> SAVE TO {self.__save_dir}')
        print('')
        # Baseline
        print("baseline (random guess) accuracy: {:.3f}".format(
            (float(self.__train_size)\
                /float(self.__train_size + self.__test_size))))


    def _prepare_model(self):
        """
        model preparation
        """
        if self.__wb_attack:
            self.__netG, self.__netD = create_models(
                latent_size=self.__latent_size,
                data_shape=self.__data_shape,
                gpu=self.__gpu,
                device=self.__device,
                alph=self.__alph,
                norm=self.__norm,
                dropout=self.__dropout
            )
            d_optimizer = torch.optim.Adam(self.__netD.parameters())
            g_optimizer = torch.optim.Adam(self.__netG.parameters())

            # Apply differential privacy
            if self.__apply_dp:
                privacy_engine = PrivacyEngine()

                if self.__sigma == -1:
                    # make_private_with_epsilon
                    print('------> make_private_with_epsilon')
                    self.__netD, d_optimizer, self.__train_loader\
                        = privacy_engine.make_private_with_epsilon(
                            module=self.__netD,
                            optimizer=d_optimizer,
                            data_loader=self.__train_loader,
                            target_epsilon=self.__eps,
                            target_delta=self.__delta,
                            epochs=self.__epochs,
                            max_grad_norm=self.__max_per_sample_grad_norm,
                        )
                else:
                    # make_private
                    print(f'------> make_private with sigma={self.__sigma}')
                    self.__netD, d_optimizer, self.__train_loader\
                        = privacy_engine.make_private(
                            module=self.__netD,
                            optimizer=d_optimizer,
                            data_loader=self.__train_loader,
                            noise_multiplier=self.__sigma,
                            max_grad_norm=self.__max_per_sample_grad_norm,
                            poisson_sampling=self.__use_poisson_sampling
                        )
            load_models(self.__netG, self.__netD, g_optimizer, d_optimizer,
                        self.__model_path)


    def _attack(self):
        thresholdList = [1, 20, 40, 80] # train:test=4:1

        cm, accuracies = mia(train_loader=self.__train_loader_1,
                             test_loader=self.__test_loader_1,
                             all_data_size=self.__train_size+self.__test_size,
                             device=self.__device,
                             netD=self.__netD,
                             thresholdList=thresholdList)
        for thresh, acc in zip(thresholdList, accuracies):
            print(f"white-box attack accuracy top {thresh}%: {acc:.2f}")
        draw_accuracy_bar(
            accuracies=accuracies,
            thresholdList=thresholdList,
            test_size=self.__test_size,
            all_size=self.__train_size+self.__test_size,
            model_type=self.__model_type,
            out_dir=self.__save_dir, is_white=self.__wb_attack)
        if self.__wb_attack:
            file_name = f'wb_results_{self.__model_type}.txt'
        else:
            file_name = f'bb_results_{self.__model_type}.txt'
        save_mia_results_as_txt(
            cm, accuracies, thresholdList,
            os.path.join(self.__save_dir, file_name))


    def exec_attack(self):
        self._set_device()
        self._load_data()
        self._print_baseline()

        # Load training and test data
        self._prepare_model()

        # Execute attack
        self._attack()
