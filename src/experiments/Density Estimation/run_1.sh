#!/bin/bash

mkdir -p logs

for KAPPA in 1500 1800 1300; do
    python train.py --loss gsssw --kappa ${KAPPA} --gpu 1 > logs/gsssw_kappa${KAPPA}.log 2>&1 &
done

wait
echo "Done."

