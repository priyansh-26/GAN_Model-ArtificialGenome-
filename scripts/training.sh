#!/bin/bash

python main.py --train --work_dir ./GAN_805_random --out_dir models/baseline  --g_learn 0.0001 --d_learn 0.0008 --epochs 16000 --save_that 1000 --norm None --ag_size 4000

python main.py --train --work_dir ./GAN_805_EAS --out_dir models/clip  --g_learn 0.0001 --d_learn 0.0008 --epochs 16000 --save_that 1000 --norm None --ag_size 4000 --apply_dp --sigma 0 -c 0.5

python main.py --train --work_dir ./GAN_2000 --out_dir models/dp  --g_learn 0.00008 --d_learn 0.00064 --epochs 16000 --save_that 1000 --dropout 0.1 --norm None --ag_size 4000 --apply_dp --sigma 0.04 -c 0.5