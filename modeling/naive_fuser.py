import torch
import torch.nn as nn
import torch.nn.functional as F


class MMFusionTransformer(nn.Module):
    def __init__(self, n_heads=12, hidden_dim=768, dropout=0.1, num_layers=2):
        super(MMFusionTransformer, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        self.img_fc = nn.Linear(256, hidden_dim)    # baseline feat (bert +resnet) 768, blip feat 256
        self.text_fc = nn.Linear(256, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim*2, nhead=n_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, img_tensor, text_tensor, labels=None):
        img_tensor = self.img_fc(img_tensor)
        text_tensor = self.text_fc(text_tensor)

        # Concatenate the transformed modalities along the sequence dimension
        mm_tensor = torch.cat([img_tensor, text_tensor], dim=1)
        output = self.encoder(mm_tensor)

        pooled_output = self.dropout(output)
        logits = self.fc(pooled_output)
        return {"logits": logits}