#!/usr/bin/env bash

# Extract features from input audio files
./venv/bin/python ./Program.py --extract-feature mfcc2

# Normalize the features to z-scores to make the train faster. Since the train file was ran multiple time,
# this part reduces the preparation time considerably (~1 hour)
./venv/bin/python ./Program.py --normalize-feature mfcc2

# Train a model with 3 convolutional layers, 2 LSTM layers and CTC loss function. The parameter defines the #epochs.
./venv/bin/python ./Program.py --train 5