import os
import dataclasses

@dataclasses.dataclass(frozen=False)
class LocalPaths:
    # work directory
    WORK_DIR: str = './work_dir/'

    def __post_init__(self):
        self.MODEL_DIR: str = os.path.join(self.WORK_DIR, 'models')
        # directories in WORK_DIR
        self.DATA_DIR: str = os.path.join(self.WORK_DIR, 'data')
        self.IMPUTE2_RESULTS_DIR: str = os.path.join(self.WORK_DIR, 'impute2_results')
        self.IMPUTATION_ANALYSIS_DIR: str = os.path.join(self.WORK_DIR, 'impute2_analysis')
        self.MEMINF_RESULTS_DIR: str = os.path.join(self.WORK_DIR, 'meminf_results')

        # directories in DATA_DIR
        self.IMPUTATION_DATA_DIR: str = os.path.join(self.DATA_DIR, 'imputation')
        self.TRAINING_DATA_DIR: str = os.path.join(self.DATA_DIR, 'training')

        # data path in TRAINING_DATA_DIR
        self.TRAINING_DATA_PATH: str = os.path.join(self.TRAINING_DATA_DIR, 'training_data.hapt')
        self.TEST_DATA_PATH: str = os.path.join(self.TRAINING_DATA_DIR, 'test_data.hapt')

        # data path in IMPUTATION_DATA_DIR
        self.IMPUTATION_TARGET_PATH: str = os.path.join(self.IMPUTATION_DATA_DIR, 'imputation_target.impute2_haps')
        self.STUDY_INDEX_PATH: str = os.path.join(self.IMPUTATION_DATA_DIR, 'study_index.pickle')
