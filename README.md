# CNNs vs Vision Transformers under Data Scarcity

Course project for *30562 – Machine Learning and Artificial Intelligence* at Bocconi University.

## Authors
- Anna Ghezzi
- Milan Ilic
- Takhmina Temirbay
- Mia Trifunovic
- Senyi Xia

## Project Goal
This project compares a small ResNet-style CNN and a parameter-matched Vision Transformer (ViT-Tiny) under different CIFAR-10 data regimes, with a focus on inductive bias and data scarcity.

## Experimental Setup
- Dataset: CIFAR-10
- Data fractions: 1%, 5%, 10%, 25%, 50%, 100%
- Optimizer: AdamW
- Scheduler: cosine annealing
- Capacity-matched models (~270k parameters)

## Repository Structure

```text
data/           dataset loading and splits
experiments/    saved checkpoints
figures/        plots and visualisations
models/         CNN and ViT implementations
report/         final paper
results/        CSV outputs
