# Label Mapping Documentation

## Target Taxonomy

The unified label set used for training and evaluation:

| Label | Description |
|-------|-------------|
| `ad_hominem` | Personal attack on the speaker rather than the argument |
| `appeal_to_authority` | Citing authority/credibility without substantive reasoning |
| `false_dilemma` | Presenting only two options when more exist |
| `false_cause` | Incorrectly attributing a causal relationship |
| `hasty_generalization` | Drawing an over-broad conclusion from limited evidence |
| `slippery_slope` | Claiming an action will trigger an unwarranted chain of consequences |
| `straw_man` | Misrepresenting an opponent's argument to make it easier to attack |
| `none` | No logical fallacy present |
| `other` | Fallacy type outside the target taxonomy (see below) |

`label_coarse` is derived:
- `fallacy` if `label_fine != none`
- `no_fallacy` if `label_fine == none`

---

## LOGIC Dataset Mapping

Original labels from Jin et al. (EMNLP 2022), 13 classes:

| Original Label | Mapped To | Rationale |
|----------------|-----------|-----------|
| `ad_hominem` | `ad_hominem` | Direct match |
| `false_causality` | `false_cause` | Renamed for consistency |
| `false_dilemma` | `false_dilemma` | Direct match |
| `faulty_generalization` | `hasty_generalization` | Synonymous |
| `fallacy_of_credibility` | `appeal_to_authority` | Credibility-based fallacies are authority appeals |
| `fallacy_of_extension` | `straw_man` | Extending an argument beyond its scope = straw man variant |
| `appeal_to_emotion` | `other` | Not in target taxonomy |
| `ad_populum` | `other` | Appeal to popularity — no direct target label |
| `circular_reasoning` | `other` | Not in target taxonomy |
| `fallacy_of_relevance` | `other` | Too broad; no clean target match |
| `equivocation` | `other` | Not in target taxonomy |
| `fallacy_of_logic` | `other` | Catch-all; not specific enough |
| `intentional` | `other` | Catch-all; not specific enough |

---

## CoCoLoFa Dataset Mapping

Original labels from CoCoLoFa (EMNLP 2024), 9 classes:

| Original Label | Mapped To | Rationale |
|----------------|-----------|-----------|
| `none` | `none` | Direct match |
| `appeal_to_authority` | `appeal_to_authority` | Direct match |
| `false_dilemma` | `false_dilemma` | Direct match |
| `hasty_generalization` | `hasty_generalization` | Direct match |
| `slippery_slope` | `slippery_slope` | Direct match |
| `appeal_to_majority` | `other` | Variant of ad populum; no target label |
| `appeal_to_nature` | `other` | Not in target taxonomy |
| `appeal_to_tradition` | `other` | Not in target taxonomy |
| `appeal_to_worse_problems` | `other` | Not in target taxonomy |

---

## Design Decisions

1. **Multi-label handling**: Neither dataset contains multi-label samples. No action needed.

2. **`other` label**: Retained for completeness. Models may optionally exclude `other` samples from training for cleaner class boundaries — document if done.

3. **`straw_man` coverage**: Only LOGIC's `fallacy_of_extension` maps here. `straw_man` is underrepresented; note this in the paper.

4. **`slippery_slope`**: Only present in CoCoLoFa. Zero samples from LOGIC.

5. **`false_cause`**: Only from LOGIC (`false_causality`). CoCoLoFa has no equivalent.
