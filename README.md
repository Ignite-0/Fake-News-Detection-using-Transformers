# Fake News Detection using RoBERTa

**Advanced fake news detection system using fine-tuned RoBERTa transformer with artifact mitigation and class balancing**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.35+-yellow.svg)](https://huggingface.co)
[![HuggingFace](https://img.shields.io/badge/himel05/fake--news--roberta-yellow)](https://huggingface.co/himel05/fake-news-roberta)

---

Fake News Detection using Transformers (RoBERTa)

This project fine-tunes a RoBERTa-base model for fake news detection using the WELFake dataset.
The workflow includes preprocessing, subsampling, tokenization, training with class-weighted loss, and evaluation using multiple metrics.

Dataset Usage

Dataset: WELFake (HuggingFace)

Total samples: 72,134

Used in this project: Only 1% (≈ 721 samples)
This was intentionally done to reduce training time, but it also creates major limitations described below.

⚠ Limitations
1. Only 1% of the data was used

Using such a small subset makes the model prone to:

Overfitting

Poor generalization

Unrealistically high metrics

Sensitivity to noise in the tiny sample

2. Duplicates were NOT removed

The WELFake dataset is known to contain many duplicate or near-duplicate articles.
Since duplicates were not cleaned, this may cause:

Train–test contamination
(Same article appears in train and test → inflated scores)

Incorrectly high F1 and accuracy due to repeated text

3. Artifact Leakage

The dataset contains stylistic and formatting patterns that make fake vs. real news easy to predict without true semantic understanding.

Examples of possible artifacts:

Specific punctuation patterns

Title formatting differences

Zero-shot label imbalance cues

Consistent writing style differences between sources

Since the model trained on raw text without feature purification, there is a high chance it learned dataset-specific artifacts rather than true fake news reasoning.

Model & Training

Model: RoBERTa-base

Token length: 512

Loss function: Weighted CrossEntropy

Training split: 60 / 20 / 20

Epochs: 4

GPU: CUDA (if available)

The model achieved high test-set performance, but due to the limitations above, results are not reliable for real-world use.

Results (on 1% subsampled data)
Metric	Score
Accuracy	~0.96
F1 (Fake)	~0.96
F1 (Real)	~0.96
PR-AUC	~0.99

⚠ These results are almost certainly inflated due to tiny dataset size, duplicates, and artifact leakage.

Recommendation

For a realistic model:

Use the full dataset

Remove duplicates & near duplicates

Shuffle and stratify properly

Consider cross-dataset generalization tests

Avoid leakage by carefully preprocessing titles, metadata, and URLs
