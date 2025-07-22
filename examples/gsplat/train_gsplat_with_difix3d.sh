SCENE_ID=neemo_mini_f0
DATA=DATA_DIR/${SCENE_ID}/gaussian_splat
DATA_FACTOR=4
CKPT_PATH=CKPT_DIR/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt # Path to the pretrained checkpoint file
OUTPUT_DIR=outputs/difix3d/gsplat/${SCENE_ID}

CUDA_VISIBLE_DEVICES=0 python3 examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA} --data_factor ${DATA_FACTOR} \
    --result_dir ${OUTPUT_DIR} --no-normalize-world-space --test_every 1 --ckpt ${CKPT_PATH}
