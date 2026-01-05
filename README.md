# NLP-based Sentiment Classifier (IMDB) - RNN, LSTM, Transformer

## Overview

This project implements NLP sentiment classifiers using machine learning and transformer-based architectures. It compares a word-level RNN/LSTM baseline with a Transformer encoder and documents results on the IMDB dataset.

## Tech stack

- Python, PyTorch, scikit-learn
- NLTK, unidecode, contractions for text normalization
- HuggingFace tokenizers (BPE) for subword tokenization
- Jupyter/Colab notebooks

## Data

- IMDB Dataset (CSV)
- Binary labels: positive and negative
- Download via gdown or place IMDB Dataset.csv next to the notebook

## Preprocessing and tokenization

### RNN/LSTM pipeline (word-level)

- Lowercase, unidecode, expand contractions, strip HTML, remove punctuation
- Stopword removal with "not" retained
- Porter stemming
- Vocabulary: top 10k tokens plus <UNK> and <PAD>
- Sequence length: 95th percentile plus padding

### Transformer pipeline (subword)

- Clean text with HTML unescape plus regex normalization
- BPE tokenizer (vocab_size=25000, min_frequency=2)
- <PAD> and <UNK> tokens, pad and truncate to max_seq_len=512
- Padding mask used in attention

## Models

- RNNClassifier: Embedding -> RNN -> dropout -> linear classifier
- LSTMClassifier: Embedding -> LSTM -> dropout -> linear classifier
- TransformerEncoder: token embedding plus positional encoding -> stacked encoder blocks
  with Multi-Head Attention, LayerNorm, and feed-forward network -> masked mean pooling
  -> classifier head

## Training and evaluation

- Train/val/test splits with scikit-learn
- Loss: CrossEntropyLoss
- Metrics: accuracy, F1, precision, recall
- Regularization: dropout, gradient clipping, early stopping
- Checkpoints saved per model (rnn.pt, lstm.pt, transformer.pt)

## Results

- Transformer superiority: While the RNN and LSTM models are capable of processing
  sequential data, they face inherent challenges with long sequences found in movie reviews. The
  Vanilla RNN is particularly susceptible to the vanishing gradient problem, limiting its ability to
  learn dependencies in long texts. Although the LSTM mitigates this through its gating mechanisms
  (forget, input, and output gates), it still suffers from sequential processing constraints that hinder parallelization and training speed. In Transformer, Multi-Head Attention plus positional encoding capture global dependencies without recurrence.
- BPE effectiveness: improves OOV handling and morphology (e.g., "unimpressed" -> "un" + "impressed").
- Quantitative success: test accuracy 91.39% and F1 0.9168, with close train/val accuracy
  (train 91.6%, val 93.4%), indicating good generalization with dropout.

## How to run

1. Open RNN_LSTM_Transformer.ipynb.
2. Ensure IMDB Dataset.csv is in the same folder (or download with the gdown cell).
3. Run preprocessing cells for the model you want (RNN/LSTM uses word vocab; Transformer uses BPE).
4. Train and evaluate; inference examples are at the end.

## Files

- RNN_LSTM_Transformer.ipynb: end-to-end training and inference for RNN/LSTM/Transformer.
- Sentiment_Analysis.pdf: project report and analysis.
