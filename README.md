# Optifyx Machine Learning Projects Repository

This repository contains two machine learning projects focused on different applications of classification techniques:

1. **Sentiment Analysis of Movie Reviews** - A natural language processing (NLP) project using logistic regression
2. **Plant Disease Classification** - A computer vision project using convolutional neural networks (CNN)

## Project 1: Sentiment Analysis with Logistic Regression

### Overview
This project implements a sentiment analysis classifier that predicts whether a movie review expresses positive or negative sentiment. The model is trained on the widely-used IMDB movie review dataset.

### Dataset
- **IMDB Movie Reviews Dataset**: A collection of 50,000 movie reviews labeled as positive or negative
- The dataset is balanced with an equal number of positive and negative reviews

### Implementation Details
- **Algorithm**: Logistic Regression
- **Text Processing**: Includes tokenization, stop word removal, and TF-IDF vectorization
- **Performance Metrics**: Accuracy, precision, recall, and F1-score


## Project 2: Plant Disease Classification with CNN

### Overview
This project develops a convolutional neural network (CNN) model to identify plant diseases from leaf images. The model can recognize 30 different diseases across 14 plant species, providing a valuable tool for agricultural monitoring and early disease detection.

### Dataset
- **Size**: Approximately 80,000 images of diseased plant leaves
- **Diversity**: Covers 30 different diseases across 14 plant species
- **Format**: RGB images of varying resolutions

### Implementation Details
- **Architecture**: Convolutional Neural Network (CNN)
- **Preprocessing**: Image resizing, normalization, and data augmentation
- **Training**: Implemented a custom archietecture to optimize accuracy
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix


## Requirements
```
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
tensorflow==2.5.0
matplotlib==3.4.2
nltk==3.6.2
```

## Future Work
- Improve sentiment analysis by implementing more advanced NLP techniques like BERT or transformers
- Extend the plant disease classifier to handle more plant species and disease types
- Develop a web interface for easy model access and usage
- Optimize models for mobile deployment to enable in-field disease detection

