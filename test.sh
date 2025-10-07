CUDA_VISIBLE_DEVICES=7 python -u inference.py \
--model ./checkpoint/model.pth \
--input /data0/konglingshun/dataset/Real_image/Image \
--output  results/Real_image \
--device cuda \

