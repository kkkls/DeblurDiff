CUDA_VISIBLE_DEVICES=7 python -u inference.py \
--model ./checkpoint/model.pth \
--input /data0/konglingshun/dataset/Real_image/Image \
--output  results_with_GoPro/Real_image \
--device cuda \

#!/bin/bash

#!/bin/bash

## 定义任务函数
#run_task() {
#    local gpu=$1
#    local input_dir=$2
#    local output_dir=$3
#
#    CUDA_VISIBLE_DEVICES=$gpu python -u inference.py \
#        --version v2 \
#        --task sr \
#        --upscale 1 \
#        --cfg_scale 1 \
#        --input "$input_dir" \
#        --output "$output_dir" \
#        --device cuda
#}
#
## 并行运行 4 个任务
#run_task 0 "/mnt/afs/users/konglingshun/dataset/Test_dataset/Realblur-image/Realblur-J/test/blur/blur" "results_with_GoPro/Realblur" &
#run_task 1 "/mnt/afs/users/konglingshun/dataset/Test_dataset/DVD/Blur" "results_with_GoPro/DVD" &
#run_task 2 "/mnt/afs/users/konglingshun/dataset/Test_dataset/GoPro/Blur" "results_with_GoPro/GoPro" &
#run_task 3 "/mnt/afs/users/konglingshun/dataset/Test_dataset/RWBI" "results_with_GoPro/RWBI" &
#
## 等待所有后台任务完成
#wait
#
#echo "All tasks completed"
#
#sleep 864000
