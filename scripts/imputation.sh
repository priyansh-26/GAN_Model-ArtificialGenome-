#!/bin/bash

# test 1KG_4000
# python test.py --work_dir ./GAN_2000/ --imputation --ref_type 1KG --ref_haps_size 4000

# test Baseline_4000
# python test.py --work_dir ./GAN_2000/ --imputation --model_dir models/samples/baseline --ag_file_name 16000_output.hapt --ref_type Baseline --ref_haps_size 4000

# test Clipping_20000
python test.py --work_dir ./GAN_2000/ --imputation --model_dir models/samples/clipping --ag_file_name 16000_output_regen.hapt --ref_type Clip --ref_haps_size 20000
