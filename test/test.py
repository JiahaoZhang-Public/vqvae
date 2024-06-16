# test_vqvae.py

import unittest
import torch
from models.vqvae import VQVAE

class TestVQVAE(unittest.TestCase):
    def test_vqvae_forward(self):
        model = VQVAE(in_channels=3, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
                      num_embeddings=512, embedding_dim=64, commitment_cost=0.25)
        x = torch.randn(1, 3, 128, 128)
        loss, recon, perplexity = model(x)
        self.assertIsNotNone(loss)
        self.assertIsNotNone(recon)
        self.assertIsNotNone(perplexity)

if __name__ == '__main__':
    unittest.main()