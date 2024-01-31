#!/bin/bash

cd /home/seungyup/Projects/embc_experiments/body2hands_base

for i in {1..27}
do
    echo "Data number: $i"
    python3 sample.py --checkpoint models/rot4/arm2wh_checkpoint_e80000_loss1.6281.pth --data_dir ./nc_data/test  --base_path ./ --data_num $i
done
