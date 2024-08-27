#!/bin/bash

cd ..
# custom config
DATA="../DATA"
TRAINER=ProMetaR

DATASET=$1
SEED=$2
GPU=3

SHOTS=16
SUB=new
CFG=vit_b16_c2_ep10_batch4_4+4ctx
LOADEP=10


for DATASET in eurosat #{"fgvc_aircraft","dtd","oxford_flowers","food101",'stanford_cars','oxford_pets','ucf101','caltech101','sun397','imagenet'}
do
    for SEED in 1 2 3
    do
        DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        DIRTRAIN=./B2N/train_base/${DIR}
        DIRTEST=./B2N/test_new/${DIR}
        if [ -d "$DIR" ]; then
            echo "Evaluating model"
            echo "Results are available in ${DIR}."
        else
            CUDA_VISIBLE_DEVICES=${GPU} python train.py \
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
            DATASET.SUBSAMPLE_CLASSES ${SUB} 
        fi
    done
done