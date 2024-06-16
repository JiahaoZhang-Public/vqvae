# VQ-VAE PyTorch 复现项目

本项目是对 VQ-VAE（Vector Quantized Variational Autoencoder）原始论文的 PyTorch 复现，旨在提供一个标准的实现框架，并应用于一个包含 5 个子类花朵数据集的实际案例中。项目包括数据预处理、模型构建、训练和测试。

## 目录结构
```
vqvae_project/
├── config.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── preprocess_data.py
├── models/
│   ├── vqvae.py
│   └── init.py
├── notebooks/
│   ├── exploration.ipynb
│   └── training.ipynb
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── tests/
│   ├── test_vqvae.py
│   └── init.py
├── requirements.txt
├── README.md
└── setup.py
```
## 配置文件

在 `config.py` 文件中设置项目所需的参数：

```python
class Config:
    DATA_DIR = 'data/raw/flowers'
    PROCESSED_DATA_DIR = 'data/processed/flowers'
    IMAGE_SIZE = (128, 128)  # 统一图像尺寸
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10

config = Config()
```
## 数据预处理

在 data/preprocess_data.py 中编写数据预处理脚本，将图像转换为统一尺寸

## 模型构建
## 模型训练