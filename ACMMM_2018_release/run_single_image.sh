export CUDA_VISIBLE_DEVICES=0;
FILENAME='1.jpg'
th ./run_single.lua --filename $FILENAME --gpu 0;
echo "Output saved to ./demo_result/"

# This requires that copying the "checkpoints" folder (from Google Drive) to ./ACMMM_2018_release/
# --gpu 1 to switch to GPU mode
# --gpu 0 to switch to CPU mode
