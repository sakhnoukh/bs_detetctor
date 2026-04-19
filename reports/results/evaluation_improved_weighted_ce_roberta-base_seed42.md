# Evaluation Report: improved_weighted_ce_roberta-base_seed42

Model dir: `/home/nlp01/bs_detector/models/improved_weighted_ce_roberta-base_seed42`
Eval split: **test**

## Metrics

- accuracy: 0.7457
- macro-F1: 0.6790
- weighted-F1: 0.7437

## Per-class F1

| Class | F1-Score |
|-------|----------|
| `false_cause` | 0.4667 |
| `straw_man` | 0.5200 |
| `hasty_generalization` | 0.5725 |
| `appeal_to_authority` | 0.6919 |
| `ad_hominem` | 0.7170 |
| `false_dilemma` | 0.7222 |
| `none` | 0.7900 |
| `other` | 0.7970 |
| `slippery_slope` | 0.8333 |

## Confusion matrix

Saved to: `/home/nlp01/bs_detector/reports/confusion_matrix_improved_weighted_ce_roberta-base_seed42.csv`

## Full classification report

```
precision    recall  f1-score   support

          ad_hominem     0.7308    0.7037    0.7170        54
 appeal_to_authority     0.6809    0.7033    0.6919        91
         false_cause     0.4667    0.4667    0.4667        30
       false_dilemma     0.6842    0.7647    0.7222        68
hasty_generalization     0.6356    0.5208    0.5725       144
                none     0.7826    0.7975    0.7900       316
               other     0.7877    0.8065    0.7970       460
      slippery_slope     0.8333    0.8333    0.8333        78
           straw_man     0.5200    0.5200    0.5200        25

            accuracy                         0.7457      1266
           macro avg     0.6802    0.6796    0.6790      1266
        weighted avg     0.7434    0.7457    0.7437      1266
```

