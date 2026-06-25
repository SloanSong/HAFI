import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, output_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim) if output_dim else None
        self.dropout = nn.Dropout(0.3)

    def forward(self, z):
        h = F.gelu(self.fc1(z))
        h = self.dropout(h)
        if self.fc2:
            return self.fc2(h)
        return h


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=256, hidden_dim=512):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, compute_loss=False):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z)
        if compute_loss:
            kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
            recon_loss = F.mse_loss(recon, x, reduction='mean')
            return z, recon, kl, recon_loss
        return z, recon, mean, logvar

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return self.reparameterize(mean, logvar)