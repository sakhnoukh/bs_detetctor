# Improved Modeling Report (focal)

Run: `improved_focal_roberta-base_seed42`
Model: `roberta-base`
Loss: **focal** with class weights
Labels: 9 classes (including `other`)
Epochs: 5 (early stopping patience=2)
Label smoothing: 0.1

## Class weights

| Class | Train count | Weight |
|-------|-------------|--------|
| `ad_hominem` | 275 | 3.180 |
| `appeal_to_authority` | 532 | 1.644 |
| `false_cause` | 204 | 4.286 |
| `false_dilemma` | 529 | 1.653 |
| `hasty_generalization` | 758 | 1.154 |
| `none` | 2201 | 0.397 |
| `other` | 2811 | 0.311 |
| `slippery_slope` | 431 | 2.029 |
| `straw_man` | 129 | 6.779 |

## Dev metrics

- accuracy: 0.7617647058823529
- macro-F1: 0.6853933420376455
- weighted-F1: 0.760842218274936

## Test metrics

- accuracy: 0.7259083728278041
- macro-F1: 0.6741125903125901
- weighted-F1: 0.7262371352033515

## Artifacts

- model + tokenizer saved to: `/home/nlp01/bs_detector/models/improved_focal_roberta-base_seed42`
- label space: `label_space.json`
- training config: `train_config.json`
