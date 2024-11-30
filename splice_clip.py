# splice_clip.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpliceSiteCLIP(nn.Module):
    def __init__(self, gena_lm, feature_dim=128):
        super().__init__()
        self.sequence_encoder = gena_lm
        
        # Freeze GENA-LM weights initially
        for param in self.sequence_encoder.parameters():
            param.requires_grad = False
        
        self.sequence_projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, feature_dim)
        )
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(5, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, feature_dim)
        )
        
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
    
    def forward(self, input_ids, attention_mask, features):
        sequence_outputs = self.sequence_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        sequence_embedding = sequence_outputs.last_hidden_state[:, 0]
        sequence_embedding = self.sequence_projector(sequence_embedding)
        
        feature_embedding = self.feature_encoder(features)
        
        sequence_embedding = F.normalize(sequence_embedding, dim=-1)
        feature_embedding = F.normalize(feature_embedding, dim=-1)
        
        return sequence_embedding, feature_embedding