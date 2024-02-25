# Neural-Machine-Translation

## Project Structure

This Neural Machine Translation (NMT) project with attention is organized into several files, each serving a specific purpose:

- **`neural_attention.py`**: Defines the custom attention mechanism used in the NMT model, allowing it to focus on relevant parts of the input sentence during translation.

- **`attention_mechanism.py`**: Implements the attention mechanism, a crucial component for aligning input and output sequences in the NMT model.

- **`BeamSearchDecoder.py`**: Implements the beam search decoding algorithm, used to generate more accurate translations by considering multiple hypotheses at once.

- **`Language.py`**: Contains the `Language` class for handling language-specific details such as vocabulary building and word-index conversions.

- **`NMT_ETL.py`**: Provides functions for the Extract, Transform, Load (ETL) process, which involves reading, preprocessing, and preparing the training data.

- **`run_evaluate.sh`**: A bash script for setting up the environment and parameters to evaluate the trained NMT model on new input sentences.

- **`run_training.sh`**: A bash script for setting up the environment and parameters to train the NMT model, specifying various hyperparameters and settings for the training process.

- **`sequence_encoder.py`**: Defines the sequence encoder component of the NMT model, which encodes the input sentence into a fixed-length context vector.

- **`topk_decoder.py`**: Implements a decoder that uses top-k sampling, an alternative to beam search for generating translations by sampling from the top k predictions at each step.

- **`train_nmt.py`**: The main training script that sets up the model, loss function, optimizer, and handles the training loop, including forward and backward passes, parameter updates, and logging.

- **`translation_model.py`**: Contains the code for the complete translation model, integrating the encoder, decoder, and attention mechanism, and includes functions for translating new sentences using the trained model.

- **`utils.py`**: Contains utility functions used across the project, such as time formatting, text normalization, and plotting the training loss.
