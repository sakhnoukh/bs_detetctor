# Evaluation Report: improved_focal_roberta-base_seed42

Model dir: `/home/nlp01/bs_detector/models/improved_focal_roberta-base_seed42`
Eval split: **test**

## Metrics

- accuracy: 0.7259
- macro-F1: 0.6741
- weighted-F1: 0.7262

## Per-class F1

| Class | F1-Score |
|-------|----------|
| `false_cause` | 0.4545 |
| `straw_man` | 0.5306 |
| `hasty_generalization` | 0.5816 |
| `ad_hominem` | 0.6981 |
| `appeal_to_authority` | 0.7047 |
| `false_dilemma` | 0.7324 |
| `other` | 0.7542 |
| `none` | 0.7755 |
| `slippery_slope` | 0.8354 |

## Confusion matrix

Saved to: `/home/nlp01/bs_detector/reports/confusion_matrix_improved_focal_roberta-base_seed42.csv`

## Full classification report

```
precision    recall  f1-score   support

          ad_hominem     0.7115    0.6852    0.6981        54
 appeal_to_authority     0.6667    0.7473    0.7047        91
         false_cause     0.4167    0.5000    0.4545        30
       false_dilemma     0.7027    0.7647    0.7324        68
hasty_generalization     0.5942    0.5694    0.5816       144
                none     0.7695    0.7816    0.7755       316
               other     0.7722    0.7370    0.7542       460
      slippery_slope     0.8250    0.8462    0.8354        78
           straw_man     0.5417    0.5200    0.5306        25

            accuracy                         0.7259      1266
           macro avg     0.6667    0.6835    0.6741      1266
        weighted avg     0.7276    0.7259    0.7262      1266
```

