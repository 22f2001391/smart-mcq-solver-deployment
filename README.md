# Smart MCQ Solver - BiLSTM Deployment

This repository contains the deployment files for the BiLSTM-based
Smart MCQ Solver.

## Model

The model is a BiLSTM implemented from scratch using PyTorch.

Architecture:

Embedding
→ 2-Layer Bidirectional LSTM
→ Masked Mean Pooling
→ Dropout
→ Linear Classifier

## Deployment

The model is deployed using:

- PyTorch
- Gradio
- Hugging Face Spaces
- ZeroGPU

## Input

The application accepts:

- MCQ prompt
- Option A
- Option B
- Option C
- Option D
- Option E

## Output

The application returns:

- Predicted answer
- Top-3 predictions with confidence scores

## Live Demo

https://huggingface.co/spaces/Ronnie-651/smart-mcq-solver-bilstm
Hugging Face Space:

https://huggingface.co/spaces/Ronnie-651/smart-mcq-solver-bilstm
