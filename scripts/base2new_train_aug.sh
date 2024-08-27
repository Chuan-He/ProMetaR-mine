#!/bin/bash

cd ..
# custom config
DATA="../DATA"
TRAINER=ProMetaR

DATASET=$1
SEED=$2
GPU=0
LOADEP=10
CFG=vit_b16_c2_ep10_dataaug

SHOTS=16

for DATASET in eurosat #{"fgvc_aircraft","dtd","oxford_flowers","food101",'stanford_cars','oxford_pets','ucf101','caltech101','sun397','imagenet'}
do
    for SEED in 1 2 3 4 5
    do
        DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        DIRTRAIN=./B2N/train_base/${DIR}
        DIRTEST=./B2N/test_new/${DIR}
        if [ -d "$DIRTRAIN" ]; then
            echo "Oops! The results exist at ${DIRTRAIN} (so skip this job)"
        else
            echo "Run this job and save the output to ${DIR}"
            CUDA_VISIBLE_DEVICES=${GPU} python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIRTRAIN} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base 
        fi

        if [ -d "$DIRTEST" ]; then
            echo "Oops! The results exist at ${DIRTEST} (so skip this job)"
        else
            CUDA_VISIBLE_DEVICES=0 python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIRTEST} \
            --model-dir ${DIRTRAIN} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES new
        fi
    done
done