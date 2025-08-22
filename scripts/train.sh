export NUM_NODES=1
export NUM_GPUS=2
accelerate launch --mixed_precision=bf16 --main_process_port 29501 --multi_gpu --num_machines $NUM_NODES --num_processes $NUM_GPUS src/train_difix.py \
    --output_dir=./outputs/neemo_mini/difix_full_tune_train \
    --root_dir="data/neemo_mini" \
    --dataset_path="data/neemo_mini/difix_pair_data.json" \
    --resolution 512 \
    --image_width 1600 \
    --image_height 900 \
    --learning_rate 2e-5 \
    --train_batch_size 2 \
    --dataloader_num_workers 8 \
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
    --report_to "tensorboard" \
    --tracker_project_name "difix_ref" \
    --tracker_run_name "0822_train_neemo_full_tune_difix_ref_model_res_900_1600" \
    --timestep 199
