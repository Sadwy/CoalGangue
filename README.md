# Install
```shell
# 系统: Ubuntu 21.04
# 创建conda环境
conda create -n mmseg python=3.8 -y
conda activate mmseg

# 配置环境包
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
pip install openmim
python -m mim install mmcv==2.0.0
python -m mim install mmengine==0.7.3

# 安装mmsegmentation==1.0.0
git clone https://github.com/Sadwy/CoalGangue.git
export MMSEG=~/CoalGangue
cd ${MMSEG}
python -m pip install -e .
```

# Dataset Pre-procession
```shell
# 安装包
pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c conda-forge matplotlib-base

# 链接数据集
ln -s dataset data

# 预处理
cd data/CoalGangue
python rgb2gray.py
python check.py  # 不报错说明预处理完成
```

# Train
```shell
python tools/train.py configs/custom/fcn_coal.py
# RTX 3090 单卡训练需要1h
```
## 1. 修改配置
- configs/custom/fcn_coal.py
    - 第13行, `max_iters` 设置迭代次数
    - 第18行, `interval` 设置保存权重文件的间隔

# Test
```shell
python tools/test.py configs/custom/fcn_coal.py work_dirs/fcn_coal/iter_10000.pth --out sadwy
```
- 使用36张验证集图像做测试, 结果图像保存在 `sadwy` 文件夹中, 但是效果几乎不可视.

# Visualization
提供了两种曲线可视化方法.
## 1. tensorboard
```shell
# 安装包
pip install tensorboardX
pip install future tensorboard
```
(此处步骤同[此链接](https://github.com/Sadwy/mlp-cnn#visualization)中README的Visualization部分.)

训练时输出的信息中有显示 `Exp name: {CONFIG_NAME}_{RUN_TIME}`, 比如 `Exp name: fcn_cocal_20230601_060606`. 则使用以下指令可视化:
```shell
tensorboard --host localhost --load_fast=true --logdir work_dirs/fcn_coal/20230601_060606
```
- 指令中的路径是训练时自动创建的. 其中`fcn_coal` 和 `20230601_060606` 分别是配置文件名称和训练时间, 视情况修改.
- 如果报错, 尝试修改指令中参数为 `--load_fast=false` (Details: [issue](https://github.com/tensorflow/tensorboard/issues/4784)). 这可能导致部分曲线显示异常.

## 2. tools/analysis_tools
```shell
# 可视化 mAcc, mFscore, mPrecision
python tools/analysis_tools/analyze_logs.py work_dirs/fcn_coal/20230601_060606/vis_data/20230601_060606.json --keys mAcc mFscore mPrecision --legend mAcc mFscore mPrecision --out curve.jpg
```
- 上述示例指令的`json`文件路径视不同配置文件和运行时间而变化.
- `--keys` 和 `--legend` 的参数应保持相同, 可选的参数有: mIoU mAcc mDice mFscore mPrecision mRecall aAcc loss
- `--out` 的参数为曲线文件名.

# Appendix
```shell
# 实现了Unet网络进行训练
# 但是奇怪地, 模型效果弱于FCN
python tools/train.py configs/custom/unet_coal.py

# 重新开始训练的指令 (不同于mmPreTrain)
# https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html#training-on-a-single-gpu
python tools/train.py configs/custom/fcn_coal.py --resume --cfg-options load_from=work_dirs/fcn_coal/iter_2000.pth
```

# Citation
```
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```
