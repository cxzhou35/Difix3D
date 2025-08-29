MODEL_PATH="checkpoints/0822_train_neemo_mini_full_difix_ref_model_res_576_1024/model_100001.pkl"
INPUT_PATH="/home/zhouchenxu/codes/Difix3D/data/neemo_mini/sf_recon/render"
REF_PATH="/home/zhouchenxu/codes/Difix3D/data/neemo_mini/sf_recon/ref"
OUTPUT_PATH="outputs/0822_mix_eval_neemo_mini_full_difix_ref_model_res_720_1280/inference"

CUDA_VISIBLE_DEVICES=$1 python3 src/inference_difix.py \
    --model_path $MODEL_PATH \
    --input_path $INPUT_PATH \
    --ref_path $REF_PATH \
    --output_path $OUTPUT_PATH \
    --data_format evc \
    --height 720 \
    --width 1280 \
    --prompt "remove degradation and artifacts" \
    --timestep 199 \
    --force
