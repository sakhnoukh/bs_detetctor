# Week 9 Evaluation Report

Model run: `finetune_roberta-base_seed42`
Model dir: `/home/nlp01/bs_detector/models/finetune_roberta-base_seed42`

## Metrics (test set)

- accuracy: 0.7543
- macro-F1: 0.6807
- weighted-F1: 0.7497

## Confusion matrix

Saved to: `/home/nlp01/bs_detector/reports/confusion_matrix_week9.csv`

## Per-class report

```
precision    recall  f1-score   support

          ad_hominem     0.6552    0.7037    0.6786        54
 appeal_to_authority     0.7191    0.7033    0.7111        91
         false_cause     0.4815    0.4333    0.4561        30
       false_dilemma     0.8065    0.7353    0.7692        68
hasty_generalization     0.6818    0.5208    0.5906       144
                none     0.7870    0.8070    0.7969       316
               other     0.7619    0.8348    0.7967       460
      slippery_slope     0.8481    0.8590    0.8535        78
           straw_man     0.6923    0.3600    0.4737        25

            accuracy                         0.7543      1266
           macro avg     0.7148    0.6619    0.6807      1266
        weighted avg     0.7511    0.7543    0.7497      1266
```

## Notes

- The label space includes `other` as a real class.
- If you want additional analysis (top confusions, error examples), extend this report after inspection.
