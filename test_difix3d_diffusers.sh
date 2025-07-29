input="/home/zhouchenxu/codes/minivolcap2/evals/20250716/143502_0715_neemo_mini_orig_sf_render/images/val/33"
ref="/home/zhouchenxu/codes/minivolcap2/evals/20250729/161037_0729_neemo_mini_orig_sf_render/images/06/gt"
output="/home/zhouchenxu/codes/Difix3D/outputs/neeomi_mini_v33_ref"

CUDA_VISIBLE_DEVICES=0 python test_difix3d_diffusers.py \
    -i $input \
    -r $ref \
    -o $output \
