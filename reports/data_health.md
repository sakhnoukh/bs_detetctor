# Data Health Report

Generated: 2026-03-08

---

## 1. Pipeline Stage Counts

| Stage | Records |
|-------|---------|
| After import (Stage 1) | 11,506 |
| After quality filter (Stage 4) | 11,458 |
| After deduplication (Stage 5) | 11,176 |
| **Final processed total** | **11,176** |

Records removed: 330 (2.9% of raw)

---

## 2. Final Split Sizes

| Split | Records |
|-------|---------|
| train | 7,851 |
| dev | 2,053 |
| test | 1,272 |
| **Total** | **11,176** |

---

## 3. Label Distribution (all splits combined)

| Label | Count | % |
|-------|-------|---|
| `ad_hominem` | 377 | 3.4% |
| `appeal_to_authority` | 742 | 6.6% |
| `false_cause` | 276 | 2.5% |
| `false_dilemma` | 738 | 6.6% |
| `hasty_generalization` | 1,098 | 9.8% |
| `none` | 3,128 | 28.0% |
| `other` | 3,984 | 35.6% |
| `slippery_slope` | 649 | 5.8% |
| `straw_man` | 184 | 1.6% |
| **Total** | **11,176** | 100% |

### Coarse label split

| Coarse Label | Count |
|--------------|-------|
| `fallacy` | 8,048 |
| `no_fallacy` | 3,128 |

---

## 4. Data Sources

| Source | Count | % |
|--------|-------|---|
| `cocolofa` | 7,702 | 68.9% |
| `logic_climate` | 1,071 | 9.6% |
| `logic_edu` | 2,403 | 21.5% |
| **Total** | **11,176** | 100% |

---

## 5. Text Length Statistics (tokens, whitespace-split)

| Stat | Value |
|------|-------|
| Min | 5 |
| Max | 284 |
| Mean | 52.5 |
| Median | 49 |

---

## 6. Label Distribution Per Split

### Train (7,851 records)

| Label | Count | % |
|-------|-------|---|
| `ad_hominem` | 273 | 3.5% |
| `appeal_to_authority` | 531 | 6.8% |
| `false_cause` | 203 | 2.6% |
| `false_dilemma` | 529 | 6.7% |
| `hasty_generalization` | 756 | 9.6% |
| `none` | 2,200 | 28.0% |
| `other` | 2,800 | 35.7% |
| `slippery_slope` | 430 | 5.5% |
| `straw_man` | 129 | 1.6% |
| **Total** | **7,851** | 100% |

### Dev (2,053 records)

| Label | Count | % |
|-------|-------|---|
| `ad_hominem` | 49 | 2.4% |
| `appeal_to_authority` | 120 | 5.8% |
| `false_cause` | 42 | 2.0% |
| `false_dilemma` | 141 | 6.9% |
| `hasty_generalization` | 197 | 9.6% |
| `none` | 611 | 29.8% |
| `other` | 722 | 35.2% |
| `slippery_slope` | 141 | 6.9% |
| `straw_man` | 30 | 1.5% |
| **Total** | **2,053** | 100% |

### Test (1,272 records)

| Label | Count | % |
|-------|-------|---|
| `ad_hominem` | 55 | 4.3% |
| `appeal_to_authority` | 91 | 7.2% |
| `false_cause` | 31 | 2.4% |
| `false_dilemma` | 68 | 5.3% |
| `hasty_generalization` | 145 | 11.4% |
| `none` | 317 | 24.9% |
| `other` | 462 | 36.3% |
| `slippery_slope` | 78 | 6.1% |
| `straw_man` | 25 | 2.0% |
| **Total** | **1,272** | 100% |
