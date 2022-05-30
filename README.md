## Introduction

In this work, we want to know to which musical genre a song belongs, given a set of attributes, such as: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness and so others.

## Summary

The model was developed based on Professor Ivanovitch's lecture notes from the Machine Learning discipline of the Graduate Program in Electrical and Computer Engineering. A complete data pipeline was built using Google Colab, Scikit-Learn and Weights & Bias to train a decision tree model. The general image of the data pipeline is shown below:

Figura 1

For the sake of understanding, a simple hyperparameter-tuning was conducted using a Random Sweep of Wandb, and the hyperparameters values adopted in the train were:

    •  full_pipeline__num_pipeline__num_transformer__model: 2
    •  classifier__criterion: 'entropy'
    •  classifier__splitter: 'best'
    •  classifier__random_state: 41

This model is used as a proof of concept for the evaluation of an entire data pipeline incorporating Machine Learning fundamentals. The data pipeline is composed of the following stages: a) fecht data, b) eda, c) preprocess, d) data check, e) segregate, f) train and g) test.

The dataset used in this project is based on the musical genre of the songs available in the Spotify application. The data was extracted from kaggle and contains information on danceability, energy, tonality, volume, mood, speech, acoustics, instrumentality, liveliness and more. The target column, or what we want to predict, is what musical genre that song belongs to, such as: Trap, Techno, Techhouse, Trance, Psytrance, Dark Trap, DnB (drums and bass), Hardstyle, Underground Rap, Trap Metal, Emo , Rap, RnB, Pop or Hiphop. The size of the dataset is about 43,205 rows and 22 columns.
