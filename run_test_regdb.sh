CUDA_VISIBLE_DEVICES=0,1,2,3 \
python test_regdb.py \
-b 256 -a agw -d  regdb_rgb \
--iters 100 \
--eps 0.6 --num-instances 16 \
--logs-dir "/data1/wzs/cvpr_upload/origin/"
