CUDA_VISIBLE_DEVICES=0,1,2,3 \
python test_sysu.py \
-b 256 -a agw -d  sysu_all \
--iters 200 \
--eps 0.6 \
--num-instances 16 \
--logs-dir "/data1/wzs/cvpr23_upload/origin"
