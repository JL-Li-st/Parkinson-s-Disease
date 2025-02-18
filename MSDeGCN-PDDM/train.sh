#!/bin/bash

# Train the model using the specified parameters
python train.py \
  --train_data_path "" \
  --train_label_path "label/STS/train_labels.pkl" \
  --val_data_path "data_processed/processed_STS_aligned_transformed_test/*.pkl" \
  --val_label_path "label/STS/test_labels.pkl" \
  --batch_size 32 \
  --num_workers 4 \
  --learning_rate 0.001 \
  --epochs 50 \
  --milestones "[30,40]" \
  --gamma 0.1 \
  --num_class 3 \
  --num_person 1 \
  --device cuda:0 \
  --seed 1