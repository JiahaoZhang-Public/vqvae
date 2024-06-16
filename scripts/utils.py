# utils.py

import os
import torch
from torchvision.utils import save_image


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    """Save model checkpoint."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    print(f'Model checkpoint saved at {checkpoint_path}')


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f'Model checkpoint loaded from {checkpoint_path}')
    return model, optimizer, checkpoint['epoch']


def save_reconstructed_images(recon_images, output_dir, epoch):
    """Save reconstructed images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_image(recon_images, os.path.join(output_dir, f'recon_epoch_{epoch}.png'))
    print(f'Reconstructed images saved at epoch {epoch}')


def calculate_perplexity(encodings):
    """Calculate the perplexity of the encodings."""
    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    return perplexity.item()


def get_device():
    """Get the available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')