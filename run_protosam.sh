#!/bin/bash
set -e
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

# Configs
MODEL_NAME='dinov2_l14' # relevant for ALPNET, aviailable: dinov2_l14, dinov2_l14_reg, dinov2_b14, dinov2_b14_reg, dlfcn_res101 (deeplabv3)
COARSE_PRED_ONLY="False" # True will output the coarse segmentation result 
PROTOSAM_SAM_VER="sam_h" # available: sam_h, sam_b, medsam
INPUT_SIZE=672 # resolution
ORGAN="rk" # relevant for MRI and CT, available: rk, lk, liver, spleen

# get modality as arg
MODALITY=$1

PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
ALL_EV=( 0 ) # 5-fold cross validation (0, 1, 2, 3, 4)
SEED=42

if [ $MODALITY != "ct" ] && [ $MODALITY != "mri" ] && [ $MODALITY != "polyp" ]
then
    echo "modality must be either ct ,mri or polyp"
    exit 1
fi

if [ $MODALITY == "ct" ]
then
    DATASET='SABS_Superpix'
fi
if [ $MODALITY == "mri" ]
then
    DATASET='CHAOST2_Superpix'
fi
if [ $MODALITY == "polyp" ]
then
    DATASET='polyps'
fi

if [ $INPUT_SIZE -gt 256 ] 
then
    DATASET=${DATASET}'_672'
fi

NWORKER=4
LORA=0
RELOAD_PATH=( "None" ) 
SKIP_SLICES="True"
DO_CCA="True"
ALL_SCALE=( "MIDDLE") # config of pseudolabels

if [ $MODALITY == "polyp" ]
then
    ORGAN="polyps"
fi

FREE_DESC=""
CPT="${MODEL_NAME}_${MODALITY}"
if [ -n "$FREE_DESC" ]
then
    CPT="${CPT}_${FREE_DESC}"
fi

if [ $LORA -ne 0 ]
then
    CPT="${CPT}_lora_${LORA}"
fi

if [ $DO_CCA = "True" ]
then
    CPT="${CPT}_cca"
fi

CPT="${CPT}_grid_${PROTO_GRID}_res_${INPUT_SIZE}_${ORGAN}_fold"

SUPP_ID='[6]' 
if [ $MODALITY == "mri" ]
then
    SUPP_ID='[4]'
fi

echo ===================================

for ((i=0; i<${#ALL_EV[@]}; i++))
do
    EVAL_FOLD=${ALL_EV[i]}
    CPT_W_FOLD="${CPT}_${EVAL_FOLD}"
    echo $CPT_W_FOLD on GPU $GPUID1
    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do
        PREFIX="test_vfold${EVAL_FOLD}"
        echo $PREFIX
        LOGDIR="./test_${MODALITY}/${CPT_W_FOLD}"

        if [ ! -d $LOGDIR ]
        then
            mkdir -p $LOGDIR
        fi

        python3 validation_protosam.py with \
            "modelname=$MODEL_NAME" \
            "base_model=alpnet" \
            "coarse_pred_only=$COARSE_PRED_ONLY" \
            "protosam_sam_ver=$PROTOSAM_SAM_VER" \
            "curr_cls=$ORGAN" \
            'usealign=True' \
            'optim_type=sgd' \
            reload_model_path=${RELOAD_PATH[i]} \
            num_workers=$NWORKER \
            scan_per_load=-1 \
            'use_wce=True' \
            exp_prefix=$PREFIX \
            'clsname=grid_proto' \
            eval_fold=$EVAL_FOLD \
            dataset=$DATASET \
            proto_grid_size=$PROTO_GRID \
            min_fg_data=1 seed=$SEED \
            save_snapshot_every=$SNAPSHOT_INTERVAL \
            superpix_scale=$SUPERPIX_SCALE \
            path.log_dir=$LOGDIR \
            support_idx=$SUPP_ID \
            lora=$LORA \
            "input_size=($INPUT_SIZE, $INPUT_SIZE)"
    done
done