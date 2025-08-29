ROOT_DIR="data/neemo_mini"
PRETRAINED_PATH="/home/zhouchenxu/codes/Difix3D/checkpoints/0826_train_neemo_mini_finetune_pretrained_difix_ref_model_res_576_1024/model_050001.pkl"
OUTPUT_DIR="outputs/0826_train_neemo_mini_finetune_pretrained_difix_ref_model_res_1280_720/eval"

CUDA_VISIBLE_DEVICES=0 bake python3 src/eval_difix.py \
    --root_dir $ROOT_DIR \
    --dataset_path="$ROOT_DIR/difix_pair_data.json" \
    --pretrained_path $PRETRAINED_PATH \
    --image_width 1280 \
    --image_height 720 \
    --mv_unet \
    --timestep 199 \
    --output_dir $OUTPUT_DIR \
    --split train \
    --dataloader_num_workers 8 \

CUDA_VISIBLE_DEVICES=1 bake python3 src/eval_difix.py \
    --root_dir $ROOT_DIR \
    --dataset_path="$ROOT_DIR/difix_pair_data.json" \
    --pretrained_path $PRETRAINED_PATH \
    --image_width 1280 \
    --image_height 720 \
    --mv_unet \
    --timestep 199 \
    --output_dir $OUTPUT_DIR \
    --split test \
    --dataloader_num_workers 8 \
