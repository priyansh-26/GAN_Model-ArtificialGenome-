import os
import pickle
import random

import pandas as pd

from ..constants import LocalPaths


class Imputation:
    """
    A class that encapsulates operations for genotype imputation
    """

    def __init__(self, args) -> None:

        self.__work_dir = args.work_dir
        self.__model_dir = args.model_dir
        self.__ref_type = args.ref_type
        self.__ref_haps_size = args.ref_haps_size
        self.__ag_file_name = args.ag_file_name

        # Variables for imputation
        ## file paths
        self.__imputation_target_path = None
        self.__map_file_path = None
        self.__legend_file_path = None
        self.__ref_file_path = None
        self.__strand_g_file_path = None
        self.__save_dir = None
        self.__out_file_path = None
        self.__sh_file_path = None
        ## settings
        self.__use_prephased_g = True
        self.__use_known_haps_g = True
        self.__use_strand_g = False
        self.__start_pos = None
        self.__end_pos = None
        self.__start_l = 0
        self.__start_t = 0
        self.__start_r = 0
        ## directory path
        self.__path = LocalPaths(WORK_DIR=self.__work_dir)


    def _set_file_path(self) -> None:
        """
        Define the paths of the files required for imputation.
        """
        # map file
        self.__map_file_path\
            = os.path.join(self.__path.IMPUTATION_DATA_DIR,
                           f'chr22.map')

        # legend file for IMPUTE2
        self.__legend_file_path\
            = os.path.join(self.__path.IMPUTATION_DATA_DIR,
                           f'chr22_IMPUTE2_format.legend')

        # imputation target haplotypes
        self.__imputation_target_path = self.__path.IMPUTATION_TARGET_PATH


        # imputation reference haplotypes
        if self.__ref_type == '1KG':
            self.__ref_file_path\
                = os.path.join(
                    self.__path.IMPUTATION_DATA_DIR,
                    f'real_refsize_{self.__ref_haps_size}.impute2_haps')
        else:
            self.__ref_file_path = os.path.join(
                self.__path.IMPUTATION_DATA_DIR,
                f'{self.__ref_type}_refsize_{self.__ref_haps_size}.impute2_haps')

        # save directory
        self.__save_dir = os.path.join(
            self.__path.IMPUTE2_RESULTS_DIR,
            f'{self.__ref_type}_{self.__ref_haps_size}'
            )

        # imputation execution script
        self.__sh_file_path = os.path.join(self.__save_dir, 'exec.sh')

        # output file
        self.__out_file_path = os.path.join(self.__save_dir, 'result.impute2')



    def _create_random_list(self, _min: int, _max: int, size: int) -> list:
        """
        Create a list containing numbers from [_min, _max] of given size

        Args:
            _min (int): minimum value
            _max (int): maximum value
            size (int): size

        Returns:
            list: created list
        """
        l = []
        while(len(l) < size):
            x = random.randint(_min, _max)
            if x not in l:
                l.append(x)
        l.sort()
        return l


    def _target_data_transform(self) -> None:
        """
        If there is no target file for imputation,
        create it from the test data.
        """
        # Read the data (test data) not used for training
        original_df = pd.read_csv(
            self.__path.TEST_DATA_PATH, header=None, sep=' ')
        original_df = original_df.iloc[:, 2:]
        original_df.columns = [x for x in range(original_df.shape[1])]
        snp_size = original_df.shape[1]

        # Number of SNPs in each region
        buffer_snps_size_all = int(snp_size / 8.)
        target_snps_size_all = snp_size - 2 * buffer_snps_size_all
        # Number of known SNPs in each region
        buffer_snps_size_known = int(buffer_snps_size_all / 10.)
        target_snps_size_known = int(target_snps_size_all / 10.)
        # Start positions of each region
        self.__start_l = 0
        self.__start_t = buffer_snps_size_all
        self.__start_r = snp_size - buffer_snps_size_all

        if not os.path.exists(self.__imputation_target_path):
            # Determine the positions to keep in the
            # three regions: left buffer, target, right buffer
            if os.path.exists(self.__path.STUDY_INDEX_PATH):
                with open(self.__path.STUDY_INDEX_PATH, 'rb') as f:
                    study_index = pickle.load(f)
                    f.close()
            else:
                buffer_l_index = self._create_random_list(
                    self.__start_l, self.__start_t-1, buffer_snps_size_known)
                target_index = self._create_random_list(
                    self.__start_t, self.__start_r-1, target_snps_size_known)
                buffer_r_index = self._create_random_list(
                    self.__start_r, snp_size-1, buffer_snps_size_known)
                study_index = buffer_l_index + target_index + buffer_r_index
                with open(self.__path.STUDY_INDEX_PATH, 'wb') as f:
                    pickle.dump(study_index, f)
                    f.close()

            # target_pos_list: Positions to keep
            legend = pd.read_csv(
                os.path.join(self.__legend_file_path),
                sep=' ')

            target_data = pd.concat([
                legend.loc[study_index].reset_index(drop=True),
                original_df.loc[:, study_index].T.reset_index(drop=True)
                ], axis=1)
            target_data.insert(0, 'dummy', 'dummy')
            target_data.to_csv(self.__path.IMPUTATION_TARGET_PATH,
                               header=None, index=False, sep=' ')


    def _make_imputation_reference(self) -> None:
        """
        Convert the data created by the model or the training dataset
        into a format suitable for imputation reference.
        """
        if not os.path.exists(self.__ref_file_path):
            if self.__ref_type == '1KG':
                # load training data
                df = pd.read_csv(
                    self.__path.TRAINING_DATA_PATH, header=None, sep=' ')
            elif self.__ref_type in ['GAN', 'Clip', 'DP']:
                # load artificial genomes
                df = pd.read_csv(
                    os.path.join(self.__model_dir, self.__ag_file_name),
                    header=None, sep=' ')

            # The size of the specified dataset must be
            # at least as large as the reference size.
            dataset_size = len(df)
            assert dataset_size >= self.__ref_haps_size, \
                'dataset size for reference should be larger than\
                    ref_haps_size.'
            index_list = self._create_random_list(
                0, dataset_size/2-1, self.__ref_haps_size/2)
            index_list_twice = []
            for i in index_list:
                index_list_twice.append(i*2)
                index_list_twice.append(i*2+1)
            # save the created reference
            ref_df = df.iloc[index_list_twice, 2:].T
            ref_df.to_csv(
                self.__ref_file_path, header=None, index=False, sep=' '
            )


    def _file_existence_check(self, filepath_list):
        """
        check if the file exists.
        """
        for filepath in filepath_list:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f'{filepath}')


    def _exec_imputation(self):
        """
        Create and execute a script file to perform imputation
        """
        # Obtain the start position and end position of the region
        legend_df = pd.read_csv(self.__legend_file_path, sep=' ')
        self.__start_pos = legend_df.iloc\
            [self.__start_t:self.__start_r].iloc[0]['position']
        self.__end_pos = legend_df.iloc\
            [self.__start_t:self.__start_r].iloc[-1]['position']

        # start imputation
        print(f'---> start imputation (save_dir: {self.__save_dir})')
        # prepare the script file
        with open(self.__sh_file_path, 'wt') as sh_file:
            sh_file.write('./modules/imputation/impute2 \\\n')

            # Use pre-phased data (distributing AA, Aa, aa into haplotypes)
            # as the imputation target
            if self.__use_prephased_g:
                sh_file.write(' -use_prephased_g \\\n')
            sh_file.write(' -m ' + self.__map_file_path + ' \\\n')
            sh_file.write(' -h ' + self.__ref_file_path + ' \\\n')
            sh_file.write(' -l ' + self.__legend_file_path + ' \\\n')
            if self.__use_known_haps_g:
                # imputation target
                sh_file.write(' -known_haps_g '\
                    + self.__imputation_target_path + ' \\\n')
            if self.__use_strand_g:
                sh_file.write(' -strand_g '\
                    + self.__strand_g_file_path + ' \\\n')
            sh_file.write(f' -int {self.__start_pos} {self.__end_pos} \\\n')
            sh_file.write(' -Ne 20000 \\\n')
            sh_file.write(' -o ' + self.__out_file_path)
        sh_file.close()

        # execute the script
        os.system(f'sh {self.__sh_file_path}')


    def main(self):
        """
        Function to perform imputation.
        Prepares data and executes if all conditions are met.
        """
        # Define paths
        self._set_file_path()

        # reference size must be greater than 0
        assert self.__ref_haps_size > 0

        # If the model type is other than 1KG, perform AG_transform
        self._target_data_transform()
        self._make_imputation_reference()

        # Check if all files are available, if available then execute
        self._file_existence_check([
            self.__map_file_path,
            self.__legend_file_path,
            self.__ref_file_path,
        ])
        if self.__use_known_haps_g:
            self._file_existence_check(
                [self.__imputation_target_path]
            )
        if self.__use_strand_g:
            self._file_existence_check(
                [self.__strand_g_file_path]
            )

        # Execute if all conditions are met
        try:
            os.makedirs(self.__save_dir, exist_ok=True)
            self._exec_imputation()
        except Exception as e:
            print(e)

