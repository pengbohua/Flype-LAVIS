import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityFusionTransformer(nn.Module):
    def __init__(self, n_modalities=2, n_heads=8, hidden_dim=768, dropout=0.1):
        super(ModalityFusionTransformer, self).__init__()
        self.n_modalities = n_modalities
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        self.modalities = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_modalities)])

        self.dropout = nn.Dropout(dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x is a list of tensors of size (batch_size, 768) for each modality

        # Transform each modality tensor
        modality_outputs = []
        for i in range(self.n_modalities):
            modality_output = self.modalities[i](x[i])
            modality_outputs.append(modality_output)

        # Concatenate the transformed modalities along the sequence dimension
        sequence_output = torch.cat(modality_outputs, dim=1)

        # Apply TransformerEncoder to the concatenated sequence
        transformer_output = self.encoder(sequence_output)

        # Apply dropout and linear layer to get the final output
        pooled_output = F.adaptive_avg_pool1d(transformer_output.permute(1, 2, 0), 1).squeeze()
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits