# vqvae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return loss, quantized, perplexity, encodings


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(num_hiddens // 2, num_hiddens, 4, 2, 1)
        self.conv3 = nn.Conv2d(num_hiddens, num_hiddens, 3, 1, 1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return self.residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens, 3, 1, 1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv_trans1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, 2, 1)
        self.conv_trans2 = nn.ConvTranspose2d(num_hiddens // 2, 3, 4, 2, 1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.residual_stack(x)
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        return x


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList(
            [ResidualLayer(in_channels, num_residual_hiddens) for _ in range(num_residual_layers)])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x) + x
        return x


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_hiddens, in_channels, 1, 1, 0)

    def forward(self, inputs):
        x = F.relu(inputs)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                 embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, 1, 1)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        loss, quantized, perplexity, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity