CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_sysu.py -b 256 -a agw -d  sysu_all \
--num-instances 16 \
--data-dir "/data0/data_wzs/SYSU-MM01-Original/SYSU-MM01" \
--logs-dir "/data1/wzs/cvpr23_upload/origin" \