#!/bin/bash
set -e
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

MODE=$1
if [ $MODE != "validation" ] && [ $MODE != "training" ]
then
    echo "mode must be  either validation or training"
    exit 1
fi

# get modality as arg
MODALITY=$2
# make sure modality is either ct or mri
if [ $MODALITY != "ct" ] && [ $MODALITY != "mri" ]
then
    echo "modality must be either ct or mri"
    exit 1
fi

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
INPUT_SIZE=256
ALL_EV=( 0 ) # 5-fold cross validation (0, 1, 2, 3, 4)
if [ $MODALITY == "ct" ]
then
    DATASET='SABS_Superpix'
else
    DATASET='CHAOST2_Superpix'
fi

if [ $INPUT_SIZE -gt 256 ] 
then
    DATASET=${DATASET}'_672'
fi

NWORKER=4
MODEL_NAME='dinov2_l14'
LORA=0
RELOAD_PATH=( "None" ) 
SKIP_SLICES="True"
DO_CCA="True"
TTT="False"
NSTEP=100000
RESET_AFTER_SLICE="True"
FINETUNE_ON_SUPPORT="False"
USE_SLICE_ADAPTER="False"
ADAPTER_LAYERS=1
CLAHE=False
ALL_SCALE=( "MIDDLE") # config of pseudolabels

LABEL_SETS=$3
EXCLU='[2,3]'

if [[ $MODALITY == "mri" && $LABEL_SETS -eq 1 ]]
then
    echo "exluding 1, 4"
    EXCLU='[1,4]' # liver(1), spleen(4)
fi

ORGANS='kidneys'
if [ $LABEL_SETS -eq 1 ]
then
    ORGANS='liver_spleen'
fi


FREE_DESC=""
CPT="${MODE}_${MODEL_NAME}_${MODALITY}"
if [ -n "$FREE_DESC" ]
then
    CPT="${CPT}_${FREE_DESC}"
fi

if [[ $TTT == "True" ]]
then
    CPT="${CPT}_ttt_nstep_${NSTEP}"
    if [ $RESET_AFTER_SLICE == "True" ]
    then
        CPT="${CPT}_reset_after_slice"
    fi
fi

if [ $USE_SLICE_ADAPTER == "True" ]
then
    CPT="${CPT}_w_adapter_${ADAPTER_LAYERS}_layers"
fi

if [ $LORA -ne 0 ]
then
    CPT="${CPT}_lora_${LORA}"
fi

if [ $CLAHE == "True" ]
then
    CPT="${CPT}_w_clahe"
fi

if [ $DO_CCA = "True" ]
then
    CPT="${CPT}_cca"
fi

CPT="${CPT}_grid_${PROTO_GRID}_res_${INPUT_SIZE}"

if [ ${EXCLU} = "[]" ]
then
    CPT="${CPT}_setting1"
else
    CPT="${CPT}_setting2"
fi

CPT="${CPT}_${ORGANS}_fold"

###### Training configs (irrelavent in testing) ######
DECAY=0.95

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=25000 # interval for saving snapshot
SEED='1234'

###### Validation configs ######
SUPP_ID='[6]' # using the additionally loaded scan as support
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

python3 $MODE.py with \
    "modelname=$MODEL_NAME" \
    'usealign=True' \
    'optim_type=sgd' \
    reload_model_path=${RELOAD_PATH[i]} \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    exp_prefix=$PREFIX \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITER \
    min_fg_data=1 seed=$SEED \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    superpix_scale=$SUPERPIX_SCALE \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    support_idx=$SUPP_ID \
    lora=$LORA \
    do_cca=$DO_CCA \
    ttt=$TTT \
    adapter_layers=$ADAPTER_LAYERS \
    use_slice_adapter=$USE_SLICE_ADAPTER \
    reset_after_slice=$RESET_AFTER_SLICE \
    "input_size=($INPUT_SIZE, $INPUT_SIZE)"
    done
done