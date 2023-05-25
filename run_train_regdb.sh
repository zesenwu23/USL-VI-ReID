CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_regdb.py -b 256 -a agw -d  regdb_rgb \
--iters 100 --num-instances 16 \
--data-dir "/data0/ReIDData/RegDB" \
--logs-dir "/data1/cvpr23_upload/origin/regdb" \
--trial 1

# trial: 1,2,3,4,5,6,7,8,9,10
