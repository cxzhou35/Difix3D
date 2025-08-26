export NUM_NODES=1
export NUM_GPUS=2

TASK_NAME="0826_train_neemo_mini_finetune_pretrained_difix_ref_model_res_576_1024"
ROOT_DIR="data/neemo_mini"
PRETRAINED_MODEL="nvidia/difix_ref"

bake accelerate launch --mixed_precision=bf16 --main_process_port 29501 --multi_gpu --num_machines $NUM_NODES --num_processes $NUM_GPUS src/train_difix.py \
    --output_dir="./outputs/neemo_mini/$TASK_NAME" \
    --root_dir=$ROOT_DIR \
    --dataset_path="$ROOT_DIR/difix_pair_data.json" \
    --resolution 512 \
    --image_width 1024 \
    --image_height 576 \
    --learning_rate 2e-5 \
    --pretrained_name $PRETRAINED_MODEL \
    --train_batch_size 1 \
    --dataloader_num_workers 8 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --max_train_steps 10000 \
    --num_training_epochs 10 \
    --checkpointing_steps 2000 \
    --eval_freq 1000 \
    --viz_freq 100 \
    --seed 42 \
    --mv_unet \
    --lambda_lpips 1.0 \
    --lambda_l2 1.0 \
    --lambda_gram 1.0 \
    --gram_loss_warmup_steps 2000 \
    --report_to "wandb" \
    --tracker_project_name "difix_ref" \
    --tracker_run_name $TASK_NAME \
    --timestep 199
