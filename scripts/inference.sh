MODEL_ITER=101001
MODEL_PATH="outputs/neemo_mini/difix_full_tune_train_multigpu/checkpoints/model_${MODEL_ITER}.pkl"
INPUT_IMAGE="data/neemo_mini/sparsev2/render/45_000108.png"
OUTPUT_DIR="outputs/neemo_mini/difix_full_tune_train_multigpu/inference/model_${MODEL_ITER}"

python src/inference_difix.py \
    --model_path $MODEL_PATH \
    --input_image $INPUT_IMAGE \
    --prompt "remove degradation" \
    --output_dir $OUTPUT_DIR \
    --timestep 199
