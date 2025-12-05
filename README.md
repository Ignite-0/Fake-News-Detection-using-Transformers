# Fake News Detection using RoBERTa

**Advanced fake news detection system using fine-tuned RoBERTa transformer with artifact mitigation and class balancing**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.35+-yellow.svg)](https://huggingface.co)
[![HuggingFace](https://img.shields.io/badge/himel05/fake--news--roberta-yellow)](https://huggingface.co/himel05/fake-news-roberta)

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)


---

##  Overview

This project implements a state-of-the-art fake news detection system using:
- **Base Model:** RoBERTa-base (125M parameters)
- **Fine-tuning:** WELFake dataset with artifact mitigation
- **Techniques:** Class weighting, proper train-test splitting
- **Deployment:** Available on HuggingFace Hub

### Why This Project?

Fake news detection is crucial for maintaining information integrity. This project addresses:
1. **Data Leakage Prevention** - Strict train-test separation
2. **Class Imbalance** - Uses weighted loss for balanced learning
3. **Real-world Applicability** - Interactive prediction interface

---

##  Dataset

### WELFake Dataset
- **Source:** [HuggingFace - davanstrien/WELFake](https://huggingface.co/datasets/davanstrien/WELFake)
- **Total Samples:** 72,134 articles
- **After Deduplication:** 62,719 unique articles
- **Classes:** 
  - REAL (0): 34,616 articles (55.5%)
  - FAKE (1): 27,791 articles (44.5%)

### Data Split
```
Training:   37,443 samples (60%)
Validation: 12,482 samples (20%)
Test:       12,482 samples (20%)
```

---

##  Model Architecture

### Base Model: RoBERTa-base

```python
Model: roberta-base
Parameters: 124,647,170
Architecture: Transformer (12 layers, 768 hidden, 12 attention heads)
Max Sequence Length: 512 tokens
Task: Binary Sequence Classification
```

### Fine-tuning Configuration

```python
Training Epochs: 4
Batch Size: 16 (train), 32 (eval)
Learning Rate: 2e-5
Warmup Ratio: 0.1
Weight Decay: 0.01
Optimizer: AdamW
LR Scheduler: Linear with warmup
```


Made with ❤️ for advancing fake news detection research
