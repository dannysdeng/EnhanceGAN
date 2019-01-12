export CUDA_VISIBLE_DEVICES=0;
FILENAME='1.jpg'
th ./run_single.lua --filename $FILENAME --gpu 0;
echo "Output saved to ./demo_result/"
