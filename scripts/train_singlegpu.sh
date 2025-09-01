TASK_NAME="0826_train_neemo_mini_finetune_pretrained_difix_ref_model_res_576_1024_single_gpu"
ROOT_DIR="data/neemo_mini"
PRETRAINED_MODEL="nvidia/difix_ref"

accelerate launch --mixed_precision=bf16 --main_process_port 29501 src/train_difix.py \
    --output_dir="./outputs/neemo_mini/$TASK_NAME" \
    --root_dir=$ROOT_DIR \
    --dataset_path="$ROOT_DIR/difix_pair_data.json" \
    --resolution 512 \
    --image_width 1024 \
    --image_height 576 \
    --learning_rate 2e-5 \
    --train_batch_size 1 \
    --dataloader_num_workers 0 \
    --pretrained_name $PRETRAINED_MODEL \
    --enable_xformers_memory_efficient_attention \
    --max_train_steps 10000 \
    --num_training_epochs 20 \
    --checkpointing_steps 1000 \
    --gradient_accumulation_steps 1 \
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
