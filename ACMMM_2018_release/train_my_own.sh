export CUDA_VISIBLE_DEVICES=1
DATA_ROOT=../train_data/myimages_good_MultiWhole \
DATA_ROOT_2=../train_data/myimages_bad_MultiWhole \
dataset=folder \
th main.lua \
2>&1 | tee myLog_my_own1.txt; \
export CUDA_VISIBLE_DEVICES=1
DATA_ROOT=../train_data/myimages_good_MultiWhole \
DATA_ROOT_2=../train_data/myimages_bad_MultiWhole \
dataset=folder \
th main_stage2.lua \
2>&1 | tee myLog_my_own2.txt

