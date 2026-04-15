# Week 8 Modeling Report

Run: `finetune_roberta-base_seed42`
Model: `roberta-base`
Labels: 9 classes (including `other`)

## Dev metrics

- accuracy: 0.7769607843137255
- macro-F1: 0.6711154938264253
- weighted-F1: 0.7718960142244363

## Test metrics

- accuracy: 0.7543443917851501
- macro-F1: 0.6807053037842075
- weighted-F1: 0.749673083635988

## Artifacts

- model + tokenizer saved to: `/home/nlp01/bs_detector/models/finetune_roberta-base_seed42`
- label space: `label_space.json`
- training config: `train_config.json`
