# SUBJECT-INDEPENDENT DATASET SPLIT REPORT

**Splitting Policy**: Subject-Independent Stratified Random Split  
**Random Seed**: `42`  
**Split Ratios**: Train 70% / Val 15% / Test 15%  
**Date**: July 2026

---

## 1. Split Partition Breakdown

```
===================================================================================
                     SUBJECT-INDEPENDENT SPLIT SUMMARY
===================================================================================
Total Unique Subjects Inspected  : 20
  - Training Set Subjects        : 14 (['s01', 's18', 's16', 's02', 's09', 's06', 's12', 's04', 's19', 's17', 's14', 's03', 's10', 's20'])
  - Validation Set Subjects      : 3 (['s05', 's13', 's08'])
  - Test Set Subjects            : 3 (['s11', 's15', 's07'])

Total Samples Partitioned        : 200
  - Training Samples (`data/train`)     : 140 (70.0%)
  - Validation Samples (`data/validation`): 30 (15.0%)
  - Test Samples (`data/test`)           : 30 (15.0%)
===================================================================================
```

---

## 2. Zero-Leakage Scientific Guarantee

Because data partitioning was executed strictly at the **Subject ID boundary**, no individual's facial features exist in both training and testing sets. This guarantees zero identity leakage during Phase 3 CNN model evaluation.
