#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Language:
    # Class attributes for special tokens and their unique indices
    START_OF_SENTENCE = '<SOS>'
    END_OF_SENTENCE = '<EOS>'
    PADDING = '<PAD>'
    UNKNOWN = '<UNK>'
    SPECIAL_TOKENS = {START_OF_SENTENCE: 0, END_OF_SENTENCE: 1, PADDING: 2, UNKNOWN: 3}

    def __init__(self, language_name):
        # Initialize the Language object with a name and preset values for various dictionaries
        self.name = language_name
        self.word2index = self.SPECIAL_TOKENS.copy()  # Copy special tokens into the word-to-index dictionary
        self.word2count = {}  # This will hold word frequencies within the corpus
        self.index2word = {index: token for token, index in self.SPECIAL_TOKENS.items()}  # Reverse mapping
        self.total_words = len(self.word2index)  # Total count of unique words, starting with special tokens

    def add_sentence(self, sentence):
        # Split the sentence into words and add each to the vocabulary
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        # If the word hasn't been seen before, add it to the dictionaries
        if word not in self.word2index:
            # Assign a unique index to the new word and update the dictionaries
            self.word2index[word] = self.total_words
            self.word2count[word] = 1
            self.index2word[self.total_words] = word
            self.total_words += 1  # Increment the word count
        else:
            # If the word exists, just update its frequency
            self.word2count[word] += 1


# In[ ]:




