#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import re
import time
import unicodedata

def display_loss_graph(loss_values):
    """
    Display a graph of the loss values over time.

    Parameters:
    - loss_values: A list of loss values to plot.
    """
    plt.figure()
    fig, ax = plt.subplots()
    # Set major ticks interval to 0.2 on the y-axis
    tick_interval = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(tick_interval)
    plt.plot(loss_values)

def format_seconds_as_time(seconds_elapsed):
    """
    Convert a time measurement in seconds to a formatted string in minutes and seconds.

    Parameters:
    - seconds_elapsed: The number of seconds.

    Returns:
    - A string formatted as 'Xm Ys' where X is minutes and Y is seconds.
    """
    minutes = math.floor(seconds_elapsed / 60)
    seconds_elapsed -= minutes * 60
    return '{}m {}s'.format(minutes, seconds_elapsed)

def calculate_elapsed_time(start_time, completion_ratio):
    """
    Calculate the elapsed and remaining time since the start, given the completion ratio.

    Parameters:
    - start_time: The start time in seconds.
    - completion_ratio: The current completion ratio of the task.

    Returns:
    - A string indicating the elapsed time and estimated remaining time.
    """
    current_time = time.time()
    elapsed_seconds = current_time - start_time
    estimated_total_seconds = elapsed_seconds / completion_ratio
    remaining_seconds = estimated_total_seconds - elapsed_seconds
    return '{} (- {})'.format(format_seconds_as_time(elapsed_seconds), format_seconds_as_time(remaining_seconds))

def sanitize_text(text):
    """
    Normalize the input string by converting to lowercase, stripping whitespace,
    and removing non-letter characters.

    Parameters:
    - text: The string to normalize.

    Returns:
    - The sanitized string.
    """
    text = convert_unicode_to_ascii(text.lower().strip())
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    return text

def convert_unicode_to_ascii(text):
    """
    Convert a Unicode string to plain ASCII characters.

    Parameters:
    - text: The Unicode string to convert.

    Returns:
    - The ASCII converted string.
    """
    normalized_chars = unicodedata.normalize('NFD', text)
    ascii_chars = [char for char in normalized_chars if unicodedata.category(char) != 'Mn']
    return ''.join(ascii_chars)

def check_language_file_availability(language_code):
    """
    Verify that the dataset file for a given language exists.

    Parameters:
    - language_code: The language code (e.g., 'eng' for English).
    """
    file_path = os.path.abspath(f'./Users/maede/{language_code}.txt')
    print(file_path)

    if not os.path.isfile(file_path):
        download_url = 'http://www.manythings.org/anki/'
        print(f"The file {language_code}.txt does not exist. Please download the dataset from '{download_url}'.")
        exit(1)

def verify_language_model_parameters(language_code):
    """
    Check that the necessary model parameter files for a language exist.

    Parameters:
    - language_code: The language code (e.g., 'eng' for English).
    """
    expected_files = [
        'attention_params_{}'.format(language_code),
        'decoder_params_{}'.format(language_code),
        'encoder_params_{}'.format(language_code)
    ]
    missing_files = [file for file in expected_files if not os.path.isfile(file)]

    if missing_files:
        print(f"Model parameters for the language '{language_code}' are missing. Train a new model first.")
        exit(1)


# In[ ]:




