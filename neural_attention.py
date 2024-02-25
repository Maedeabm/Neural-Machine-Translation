#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
from torch import nn
from torch.nn import functional as F

class CustomAttention(nn.Module):
    """Custom Attention module for computing alignment scores."""

    def __init__(self, strategy, size):
        super(CustomAttention, self).__init__()
        self.strategy = strategy
        self.size = size

        # Setup layers
        self.attn_layers = nn.ModuleDict({
            'general': nn.Linear(size, size),
            'concat': nn.Linear(size * 2, size)
        })
        if strategy == 'concat':
            self.v = nn.Parameter(torch.rand(size))

    def calculate_score(self, decoder_hidden, encoder_output):
        """Determine the encoder output's importance in relation to the decoder hidden state."""
        if self.strategy == 'dot':
            return torch.dot(decoder_hidden.view(-1), encoder_output.view(-1))
        elif self.strategy == 'general':
            transformed = self.attn_layers['general'](encoder_output)
            return torch.dot(decoder_hidden.view(-1), transformed.view(-1))
        elif self.strategy == 'concat':
            combined = torch.cat((decoder_hidden, encoder_output), dim=1)
            transformed = self.attn_layers['concat'](combined)
            return torch.dot(self.v.view(-1), transformed.view(-1))

    def forward(self, decoder_hidden, encoder_outputs):
        """Apply attention to all encoder inputs based on the decoder's previous hidden state."""
        batch_sz, _ = decoder_hidden.size()
        seq_len, _, _ = encoder_outputs.size()
        attn_energies = torch.zeros(batch_sz, seq_len, device=encoder_outputs.device)

        for b in range(batch_sz):
            for l in range(seq_len):
                attn_energies[b, l] = self.calculate_score(decoder_hidden[b], encoder_outputs[l, b])

        attn_weights = F.softmax(attn_energies, dim=1).view(batch_sz, 1, seq_len)
        return attn_weights

# Example usage:
# attention = CustomAttention('general', 256)
# hidden_state = torch.randn(1, 256)
# encoder_outputs = torch.randn(10, 1, 256)
# attention_weights = attention(hidden_state, encoder_outputs)


# In[ ]:




