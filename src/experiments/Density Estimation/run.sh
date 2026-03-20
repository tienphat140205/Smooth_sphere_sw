#!/bin/bash

mkdir -p logs

for KAPPA in 700 1000 1300; do
    python train.py --loss gsssw --kappa ${KAPPA} --gpu 0 > logs/gsssw_kappa${KAPPA}.log 2>&1 &
done

wait
echo "Done."
