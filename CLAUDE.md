# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains solutions to the Manning Live Project "Sentiment Analysis and Natural Language Processing for Marketing". The project implements sentiment analysis on Amazon video game reviews using multiple approaches:

1. **Part 1**: Data preparation and corpus creation
2. **Part 2**: Dictionary-based sentiment analysis using NLTK
3. **Part 3**: Evaluation of dictionary-based analyzer
4. **Part 4**: Neural network approaches using transformer models (BERT, DistilBERT, RoBERTa)

The project dates from 2020 and uses BERT-era transformer models. The code is organized as both Python scripts and Jupyter notebooks.

## Environment Setup

The project uses Python 3.13.9 with conda and pip for package management.

### Creating the environment from scratch:

```bash
conda create -n growth-hacking-sentiment python=3.13 numpy pandas
conda activate growth-hacking-sentiment
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
conda install scipy matplotlib
conda install scikit-learn
pip install -r growth-hacking-sentiment/requirements.txt
```

### Creating from environment file:

```bash
conda env create -f environment.yml
```

### Activating the environment:

```bash
conda activate growth-hacking-sentiment
```

## Data Setup

The project requires the Amazon Video Games review dataset:
- Download from: https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz
- Place in: `growth-hacking-sentiment/data/Video_Games_5.json`
- The data file is not included in the repository

## Running the Code

All Python scripts should be run from their respective directories (part-1, part-2, part-3, part-4) as they use relative paths for data access.

### Part 1 - Data Preparation:
```bash
cd growth-hacking-sentiment/part-1
python prep-data.py
```
This creates:
- `small_corpus.csv` - Balanced dataset with 4,500 reviews
- `large_corpus.csv` - Random sample of 100,000 reviews
- Charts showing rating distributions

### Part 2 - Dictionary-Based Analysis:
```bash
cd growth-hacking-sentiment/part-2
python dictionary-based-analyser.py
```
Implements sentiment scoring using NLTK's opinion lexicon with negation handling.

### Part 3 - Evaluation:
```bash
cd growth-hacking-sentiment/part-3
python evaluating-dictionary-based-analyzer.py
```
Evaluates dictionary-based approach using accuracy, precision, recall, and confusion matrices.

### Part 4 - Neural Networks:
```bash
cd growth-hacking-sentiment/part-4
python neural_networks.py
```
This script contains multiple models (commented out in main). Uncomment the desired model:
- `eval_model1()` - Pre-trained DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- `create_model2()` + `eval_model_2()` - Fine-tuned RoBERTa classifier
- `create_model3()` + `eval_model_3()` - Fine-tuned RoBERTa with additional language model training

## Architecture

### Data Flow:
1. Raw JSON data → `prep-data.py` → Creates balanced and large corpora
2. Small corpus → Dictionary-based analyzer → Scored corpus
3. Scored corpus → Evaluation scripts → Metrics and visualizations
4. Small corpus → Neural network training → Trained models → Predictions

### Key Data Transformations:
- **Ratings to classes**: Ratings 1-5 are converted to three classes:
  - `negative`: rating <= 1
  - `neutral`: rating 2-4
  - `positive`: rating >= 5
- **Sentiment scores**: Continuous scores converted to classes using thresholds (e.g., < -0.2, -0.2 to 0.2, > 0.2)

### Model Storage:
- Part 4 creates model checkpoints in `outputs/` and `models/` directories
- These directories are created automatically during training
- Models can be large (hundreds of MB to several GB)

### Charts:
- Charts are saved as HTML files using Altair in `growth-hacking-sentiment/charts/`
- Each part has its own subdirectory (part-1, part-2)

## Key Technologies

- **NLTK**: Dictionary-based sentiment analysis, tokenization, opinion lexicon
- **PyTorch**: Backend for transformer models
- **Transformers (HuggingFace)**: Pre-trained models and pipelines
- **SimpleTransformers**: Wrapper for easier transformer model training
- **Scikit-learn**: Metrics, train/test splits, evaluation
- **Pandas**: Data manipulation
- **Altair**: Interactive visualizations
- **Matplotlib**: Confusion matrix displays

## Important Constants

- `RANDOM_SEED = 42` - Used throughout for reproducibility
- `LARGE_DATASET_SIZE = 100000` - Size of large corpus
- Max sequence length for transformers: 512 tokens (with sliding window enabled)

## Performance Notes

- Part 4 neural network training is computationally intensive
- The code detects and uses MPS (Apple Silicon) or CUDA when available
- Training times vary significantly: DistilBERT fine-tuning takes ~1-3 hours, RoBERTa with 512k sequences takes ~1 hour on GPU
- Model evaluation includes timing decorators (`@timer`) to track performance

## Model Performance Reference

Best results from Part 4 (on test set):
- **Pre-trained DistilBERT**: 64% accuracy
- **Fine-tuned RoBERTa (128 seq)**: 73% accuracy
- **Fine-tuned RoBERTa (512 seq)**: 77% accuracy
- **Fine-tuned + Language Model RoBERTa**: 75% accuracy

Dictionary-based approach (Part 2/3): 39% accuracy
