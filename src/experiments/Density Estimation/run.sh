#!/bin/bash

mkdir -p logs

# Available datasets: "fire", "flood", "quakes_all", "volerup"
DATASET="flood"

for KAPPA in 700 1000 1300; do
    python train.py --loss gsssw --dataset ${DATASET} --kappa ${KAPPA} --gpu 0 > logs/gsssw_${DATASET}_kappa${KAPPA}.log 2>&1 &
done

wait
echo "Done."
