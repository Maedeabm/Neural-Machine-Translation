#!/usr/bin/env python
# coding: utf-8

# ## ETL 
# 
# This is the ETL (Extract, Transform, Load) process to understand how data is read, preprocessed, and prepared for training.

# In[26]:


import helpers
import torch
from spacy.language import Language

# Constants
MAX_LENGTH = 20

# Functions for data extraction and preprocessing
def is_sentence_pair_short(pair):
    """Check if both sentences in the pair are shorter than MAX_LENGTH."""
    return all(len(sentence.split()) < MAX_LENGTH for sentence in pair)

def filter_short_sentence_pairs(pairs):
    """Keep only pairs where both sentences are shorter than MAX_LENGTH."""
    return [pair for pair in pairs if is_sentence_pair_short(pair)]

def load_and_prepare_data(language_code):
    """
    Load sentence pairs from file, filter, and index words.
    
    Args:
        language_code (str): Abbreviation of the target language.
    
    Returns:
        Tuple[Language, Language, List[List[str]]]: Input and output Language instances, and list of sentence pairs.
    """
    # Load sentence pairs from file
    input_lang, output_lang, pairs = read_sentence_pairs(language_code)
    
    # Filter out long sentence pairs
    pairs = filter_short_sentence_pairs(pairs)
    
    # Index words in each sentence
    for english_sentence, foreign_sentence in pairs:
        input_lang.index_words(english_sentence)
        output_lang.index_words(foreign_sentence)
    
    return input_lang, output_lang, pairs

def read_sentence_pairs(language_code):
    """Read and normalize sentence pairs from a text file."""
    file_path = f'./data/{language_code}.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split('\n')
    
    pairs = [[helpers.normalize_string(sentence) for sentence in line.split('\t')] for line in lines]
    input_lang = Language('eng')
    output_lang = Language(language_code)
    
    return input_lang, output_lang, pairs

# Functions for data transformation
def indexes_from_sentence(lang, sentence):
    """Convert a sentence to a list of word indexes."""
    return [lang.word2index[word] for word in sentence.split()]

def tensor_from_sentence(lang, sentence, device='cpu'):
    """Convert a sentence to a tensor of word indexes."""
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(Language.EOS_TOKEN)  # Append End Of Sentence token
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensor_from_pair(pair, input_lang, output_lang, device='cpu'):
    """Convert a pair of sentences to a pair of tensors."""
    input_tensor = tensor_from_sentence(input_lang, pair[0], device)
    target_tensor = tensor_from_sentence(output_lang, pair[1], device)
    return input_tensor, target_tensor


# In[ ]:




