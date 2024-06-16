# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vqvae import VQVAE
from config import config
from scripts.utils import save_checkpoint, get_device


def train(model, dataloader, optimizer, num_epochs=10):
    model.train()
    device = get_device()
    for epoch in range(num_epochs):
        total_loss = 0
        total_perplexity = 0
        for images, _ in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            loss, recon, perplexity = model(images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_perplexity += perplexity.item()

        avg_loss = total_loss / len(dataloader)
        avg_perplexity = total_perplexity / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}')
        save_checkpoint({'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'epoch': epoch}, config.CHECKPOINT_DIR)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(config.PROCESSED_DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    device = get_device()
    model = VQVAE(in_channels=3, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
                  num_embeddings=512, embedding_dim=64, commitment_cost=0.25).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train(model, dataloader, optimizer, num_epochs=config.NUM_EPOCHS)