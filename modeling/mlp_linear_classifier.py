import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, hidden_dim=768, dropout=0.1):
        super(MLPClassifier, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, multi_tensor, labels=None):
        # Concatenate the transformed modalities along the sequence dimension
        pooled_output = self.dropout(multi_tensor)
        logits = self.fc(pooled_output)
        return {"logits": logits}