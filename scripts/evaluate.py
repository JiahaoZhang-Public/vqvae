# evaluate.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vqvae import VQVAE
from config import config
from scripts.utils import load_checkpoint, get_device, save_reconstructed_images


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_perplexity = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            loss, recon, perplexity = model(images)
            total_loss += loss.item()
            total_perplexity += perplexity.item()

    avg_loss = total_loss / len(dataloader)
    avg_perplexity = total_perplexity / len(dataloader)
    print(f'Average Loss: {avg_loss:.4f}, Average Perplexity: {avg_perplexity:.4f}')

    return avg_loss, avg_perplexity


def visualize_reconstruction(model, dataloader, device, num_images=5):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        _, recon, _ = model(images)

    images = images.cpu().numpy()
    recon = recon.cpu().numpy()

    fig, axs = plt.subplots(2, num_images, figsize=(15, 3))
    for i in range(num_images):
        axs[0, i].imshow(images[i].transpose(1, 2, 0) * 0.5 + 0.5)
        axs[0, i].axis('off')
        axs[1, i].imshow(recon[i].transpose(1, 2, 0) * 0.5 + 0.5)
        axs[1, i].axis('off')
    plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(config.PROCESSED_DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    device = get_device()
    model = VQVAE(in_channels=3, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
                  num_embeddings=512, embedding_dim=64, commitment_cost=0.25).to(device)

    checkpoint_path = f"{config.CHECKPOINT_DIR}/{config.CHECKPOINT_FILE}"
    model, _, _ = load_checkpoint(checkpoint_path, model)

    evaluate(model, dataloader, device)
    visualize_reconstruction(model, dataloader, device)