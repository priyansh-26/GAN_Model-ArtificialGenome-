#!/bin/bash
# sh ./scripts/membership_inference.shで実行

python main.py --work_dir ./GAN_2000 --wb_attack --model_dir models/samples/baseline --model_name baseline.pt --model_type Baseline --dropout 0.1

# python main.py --work_dir ./GAN_2000 --wb_attack --model_dir models/samples/clipping --model_name clipping.pt --model_type Clipping --dropout 0.1 --apply_dp --sigma 0 -c 0.5

# python main.py --work_dir ./GAN_2000 --wb_attack --model_dir models/samples/dp --model_name dp.pt --model_type DP --dropout 0.1 --apply_dp --sigma 0 -c 0.5

# python main.py --work_dir ./GAN_805_random --wb_attack --model_dir models/samples/dp --model_name dp.pt --model_type DP --apply_dp --sigma 0 -c 0.5