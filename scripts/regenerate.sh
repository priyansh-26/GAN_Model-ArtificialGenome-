#!/bin/bash

python main.py --regenerate --work_dir ./GAN_805_random --model_dir models/samples/dp --model_name dp.pt --ag_size 10000

python main.py --regenerate --work_dir ./GAN_2000 --model_dir models/samples/clipping --model_name clipping.pt --ag_size 10000