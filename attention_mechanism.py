#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    """
    An attention mechanism module that computes alignment scores between the encoder
    outputs and the current hidden state of the decoder.
    """

    def __init__(self, attention_type, hidden_size):
        """
        Initialize the attention mechanism.

        Parameters:
        - attention_type: The type of attention mechanism ('dot', 'general', or 'concat').
        - hidden_size: The size of the hidden states.
        """
        super(AttentionMechanism, self).__init__()
        self.attention_type = attention_type
        self.hidden_size = hidden_size

        # Define layers based on the attention type
        if attention_type == 'general':
            self.attention_layer = nn.Linear(hidden_size, hidden_size)
        elif attention_type == 'concat':
            self.attention_layer = nn.Linear(hidden_size * 2, hidden_size)
            self.context_vector = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Compute the attention scores and return the weighted sum of encoder outputs.

        Parameters:
        - decoder_hidden: The current hidden state of the decoder.
        - encoder_outputs: The outputs from the encoder.

        Returns:
        - The context vector, which is a weighted sum of the encoder outputs.
        """
        # Initialize a tensor to store the attention energies
        batch_size = encoder_outputs.size(1)
        sequence_length = encoder_outputs.size(0)
        attention_energies = torch.zeros(batch_size, sequence_length).to(encoder_outputs.device)

        # Compute the attention energies for each encoder output
        for i in range(batch_size):
            for j in range(sequence_length):
                attention_energies[i, j] = self._calculate_energy(decoder_hidden[i], encoder_outputs[j, i])

        # Normalize the energies and reshape them to [batch_size, 1, sequence_length]
        attention_weights = F.softmax(attention_energies, dim=1).unsqueeze(1)
        return attention_weights

    def _calculate_energy(self, decoder_hidden, encoder_output):
        """
        Calculate the energy score between a decoder hidden state and an encoder output.

        Parameters:
        - decoder_hidden: A single hidden state from the decoder.
        - encoder_output: A single output from the encoder.

        Returns:
        - The energy score as a scalar.
        """
        if self.attention_type == 'dot':
            energy = torch.dot(decoder_hidden.view(-1), encoder_output.view(-1))
        elif self.attention_type == 'general':
            energy = self.attention_layer(encoder_output)
            energy = torch.dot(decoder_hidden.view(-1), energy.view(-1))
        elif self.attention_type == 'concat':
            energy = self.attention_layer(torch.cat((decoder_hidden, encoder_output), dim=1))
            energy = torch.dot(self.context_vector.view(-1), energy.view(-1))
        return energy


# In[ ]:




