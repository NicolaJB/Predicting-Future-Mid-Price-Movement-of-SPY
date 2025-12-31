# Predicting Future Mid-Price Movement of SPY  
## Classical Machine Learning Baselines on Historical Daily Returns

## Overview

This project explores short-horizon price direction prediction for the SPY stock index using classical machine learning methods applied to historical daily returns. The focus is on building methodologically correct baselines for time-series classification rather than optimising financial performance.

Two modelling pipelines are implemented and compared. The first uses a neural autoencoder for nonlinear representation learning followed by a multi-layer perceptron classifier. The second uses principal component analysis followed by a support vector machine as a fast and interpretable benchmark.

The notebook is designed as a self-contained, reproducible example suitable for machine learning portfolio review.


## Key Characteristics

- End-to-end supervised learning pipeline on real financial time-series data  
- Strict chronological train–test split to avoid lookahead bias  
- Rolling window feature construction with correct indexing  
- Feature scaling and dimensionality reduction fitted only on training data  
- Multiple evaluation metrics beyond accuracy  
- Simple backtest to connect predictions with downstream decision-making  

## Data Description

The dataset consists of historical daily prices for the SPY exchange-traded fund, downloaded using the yfinance API. Daily returns are computed from adjusted close prices where available.

Each sample is constructed using a rolling window of the past ten daily returns. The target label indicates whether the following day’s return is positive or negative.

## Methodology

### Feature Engineering

Daily percentage returns are calculated and converted into fixed-length rolling windows. Labels are generated using the next-day return to ensure a strictly causal setup.

The dataset is split chronologically into training and test sets. All scaling and representation learning steps are fitted exclusively on the training data and then applied to the test set.

### Models

#### Autoencoder to MLP

A shallow feed-forward autoencoder is trained to reconstruct rolling return windows. The encoder compresses the input into a four-dimensional latent representation. These latent features are then passed to a multi-layer perceptron classifier for binary classification.

This pipeline demonstrates nonlinear feature extraction followed by a conventional supervised classifier.

#### PCA to SVM Baseline

Principal component analysis is applied to the scaled training data to obtain a four-dimensional linear representation. A support vector machine with an RBF kernel is trained on these components and evaluated on the test set.

This serves as a strong classical baseline against which the autoencoder-based approach can be compared.

## Evaluation

Model performance is evaluated using accuracy, precision, recall and F1 score. Metrics are reported on the test set, with training metrics included where relevant to diagnose overfitting.

This multi-metric evaluation reflects best practice for imbalanced or noisy classification problems.

## Backtest Simulation

A simple backtest is included to provide intuition about how classification outputs translate into sequential decision-making. The strategy invests in SPY on days where the model predicts an upward move and remains in cash otherwise.

The resulting portfolio value is plotted over the test period. The simulation excludes transaction costs, slippage and risk constraints and is intended for illustrative purposes only.

---

## Installation and Requirements

The notebook is designed to run in Google Colab.

Required packages can be installed with:

```bash
pip install yfinance scikit-learn tensorflow matplotlib seaborn
```

### How to Run
- Open the notebook in Google Colab
- Run all cells sequentially
- Plots, metrics and backtest results are generated inline

### Limitations
- Financial time series are noisy and weakly predictive at short horizons. Results should be interpreted as a demonstration of modelling practice rather than evidence of a deployable trading strategy.
- The backtest omits transaction costs and risk management and should not be considered realistic performance estimation.

### Potential Extensions
- Incorporating transaction costs and risk-adjusted metrics
- Comparison against naive directional baselines
- Regression-based return prediction
- Regime-aware or volatility-conditioned models

### License
MIT License
