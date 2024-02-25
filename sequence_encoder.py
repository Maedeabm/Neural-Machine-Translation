#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

class SequenceEncoder(nn.Module):
    """
    This class implements an encoder for a sequence-to-sequence neural network model,
    which is used to convert a sequence of word indices (like a sentence) into a
    context vector that captures the essence of the sequence.
    """

    def __init__(self, vocabulary_size, embed_size, hidden_units, num_layers=1, dropout_rate=0.1):
        """
        Initialize the encoder with the required layers and dimensions.
        
        Parameters:
        - vocabulary_size: Number of unique words in the source language.
        - embed_size: Size of the embedding vectors.
        - hidden_units: Number of units in the RNN's hidden layers.
        - num_layers: Number of stacked RNN layers.
        - dropout_rate: Proportion of units to drop out to prevent overfitting.
        """
        super(SequenceEncoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embed_size = embed_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        # Layer definitions
        self.embedding_layer = nn.Embedding(vocabulary_size, embed_size)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.rnn_layer = nn.GRU(embed_size, hidden_units, num_layers)

    def forward(self, sequence_input, initial_hidden):
        """
        Forward pass through the encoder: embedding -> dropout -> RNN.
        
        Parameters:
        - sequence_input: A batch of input sequences.
        - initial_hidden: The initial hidden state for the RNN.
        
        Returns:
        - RNN output and the final hidden state.
        """
        # Reshape the input to [sequence_length, 1] for batch size of 1
        sequence_input = sequence_input.view(-1, 1)
        
        # Embed the input sequence
        embedded_sequence = self.embedding_layer(sequence_input)
        
        # Apply dropout for regularization
        dropped_out_sequence = self.dropout_layer(embedded_sequence)
        
        # Pass the processed sequence through the RNN
        rnn_output, updated_hidden = self.rnn_layer(dropped_out_sequence, initial_hidden)
        
        return rnn_output, updated_hidden

    def initialize_hidden_state(self, computing_device):
        """
        Initialize the hidden state to zeros before processing a new sequence.
        
        Parameters:
        - computing_device: The device (CPU/GPU) to create the hidden state on.
        
        Returns:
        - The initial hidden state for the RNN.
        """
        # Create a tensor of zeros with shape [num_layers, batch_size, hidden_units]
        initial_hidden = torch.zeros(self.num_layers, 1, self.hidden_units).to(computing_device)
        
        return initial_hidden

