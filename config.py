# config.py

class Config:
    DATA_DIR = 'data/raw/flowers'
    PROCESSED_DATA_DIR = 'data/processed/flowers'
    IMAGE_SIZE = (128, 128)  # 统一图像尺寸
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    CHECKPOINT_DIR = 'checkpoints'
    CHECKPOINT_FILE = 'vqvae_checkpoint.pth'
    OUTPUT_DIR = 'outputs'

config = Config()