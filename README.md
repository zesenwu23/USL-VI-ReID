# Unsupervised Visible-Infrared Person Re-Identification via Progressive Graph Matching and Alternate Learning


## Dataset Preprocessing
Convert the dataset format (like Market1501).
```shell
python prepare_sysu.py   # for SYSU-MM01
python prepare_regdb.py  # for RegDB
```
You need to change the file path in the `prepare_sysu(regdb).py`.

Note: a pre-processed dataset can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1Ovc8SRbWHkMMit26DfEaiA) (Password: ReID) or [Google Drive](https://drive.google.com/drive/folders/1TJG3TRgqi_DUMItJeFU4285IaB10-cXl?usp=sharing).

## Training
```shell
./train_sysu.sh   # for SYSU-MM01
./train_regdb.sh  # for RegDB
```
Two training stages are included and you need to specify the training stage by commenting another stage's `main_worker` like this:
```python
main_worker_stage1(args,log_s1_name) # Stage 1
# main_worker_stage2(args,log_s1_name,log_s2_name) # Stage 2
```


## Test
```shell
./test_sysu.sh    # for SYSU-MM01
./test_regdb.sh   # for RegDB
```

# Citation
```bibtex
@InProceedings{Wu_2023_CVPR,
    author    = {Wu, Zesen and Ye, Mang},
    title     = {Unsupervised Visible-Infrared Person Re-Identification via Progressive Graph Matching and Alternate Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9548-9558}
}
```

Our trained models can be downloaded [here](https://drive.google.com/drive/folders/1NIpM5uv9_DUbCafwy7Z28yXPnMXxNtss?usp=sharing).

# Contact
zesenwu@whu.edu.cn

The code is implemented based on ClusterContrast and ADCA.
