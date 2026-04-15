# Data Health Report

Generated: 2026-03-09

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
| train | 7,870 |
| dev | 2,040 |
| test | 1,266 |
| **Total** | **11,176** |

---

## 3. Label Distribution (all splits combined)

| Label | Count | % |
|-------|-------|---|
| `ad_hominem` | 377 | 3.4% |
| `appeal_to_authority` | 743 | 6.6% |
| `false_cause` | 274 | 2.5% |
| `false_dilemma` | 738 | 6.6% |
| `hasty_generalization` | 1,098 | 9.8% |
| `none` | 3,128 | 28.0% |
| `other` | 3,985 | 35.7% |
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

## 5. Before/After Counts Per Source (Stage 4 quality filter)

| Source | Before quality filter | After quality filter | Removed |
|--------|------------------------|----------------------|---------|
| `cocolofa` | 7,706 | 7,706 | 0 |
| `logic_climate` | 1,351 | 1,309 | 42 |
| `logic_edu` | 2,449 | 2,443 | 6 |
| **Total** | **11,506** | **11,458** | **48** |

### Stage 4 Drop Reason Breakdown

| Reason | Count |
|--------|-------|
| `empty` | 39 |
| `too_short` | 9 |
| **Total** | **48** |

---

## 6. Deduplication Summary (Stage 5)

- Retention policy: prefer train > dev > test when duplicate text appears across files
- Input records: 11,458
- Kept records: 11,176
- Dropped (same file): 258
- Dropped (cross file): 24
- Dropped (cross split): 24

---

## 7. Text Length Statistics (tokens, whitespace-split)

| Stat | Value |
|------|-------|
| Min | 5 |
| Max | 284 |
| Mean | 52.5 |
| Median | 49 |

---

## 8. Label Distribution Per Split

### Train (7,870 records)

| Label | Count | % |
|-------|-------|---|
| `ad_hominem` | 275 | 3.5% |
| `appeal_to_authority` | 532 | 6.8% |
| `false_cause` | 204 | 2.6% |
| `false_dilemma` | 529 | 6.7% |
| `hasty_generalization` | 758 | 9.6% |
| `none` | 2,201 | 28.0% |
| `other` | 2,811 | 35.7% |
| `slippery_slope` | 431 | 5.5% |
| `straw_man` | 129 | 1.6% |
| **Total** | **7,870** | 100% |

### Dev (2,040 records)

| Label | Count | % |
|-------|-------|---|
| `ad_hominem` | 48 | 2.4% |
| `appeal_to_authority` | 120 | 5.9% |
| `false_cause` | 40 | 2.0% |
| `false_dilemma` | 141 | 6.9% |
| `hasty_generalization` | 196 | 9.6% |
| `none` | 611 | 30.0% |
| `other` | 714 | 35.0% |
| `slippery_slope` | 140 | 6.9% |
| `straw_man` | 30 | 1.5% |
| **Total** | **2,040** | 100% |

### Test (1,266 records)

| Label | Count | % |
|-------|-------|---|
| `ad_hominem` | 54 | 4.3% |
| `appeal_to_authority` | 91 | 7.2% |
| `false_cause` | 30 | 2.4% |
| `false_dilemma` | 68 | 5.4% |
| `hasty_generalization` | 144 | 11.4% |
| `none` | 316 | 25.0% |
| `other` | 460 | 36.3% |
| `slippery_slope` | 78 | 6.2% |
| `straw_man` | 25 | 2.0% |
| **Total** | **1,266** | 100% |
