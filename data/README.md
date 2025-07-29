# Experiment Data Directory

This directory contains experimental results from the thinking transplant study.

## Directory Structure

```
data/
├── phase1/          # Phase 1 results
├── phase2/          # Phase 2 results  
├── phase3/          # Phase 3 results (future)
└── analysis/        # Combined analysis
```

## File Naming

Files use timestamps for easy identification:

```
phase1_thinking-experiment_YYYYMMDD_HHMMSS.csv
phase2_transplant-test_YYYYMMDD_HHMMSS.csv
```

## File Contents

**CSV Files**: Complete trial data with columns for model_name, condition, problem_id, response, digits_correct, and error information.

**Analysis**: Load files with pandas and group by condition to compare accuracy across experiments.

## Data Analysis Examples

```python
import pandas as pd

# Load Phase 1 results
df1 = pd.read_csv('data/phase1/phase1_thinking-experiment_20250729_230045.csv')
accuracy = df1.groupby('condition')['digits_correct'].mean()

# Load Phase 2 results  
df2 = pd.read_csv('data/phase2/phase2_transplant-test_20250729_230045.csv')
baseline = df2[df2['condition'] == 'baseline_no_numbers']['digits_correct'].mean()
transplant = df2[df2['condition'] == 'with_transplanted_numbers']['digits_correct'].mean()
effect = transplant - baseline
```

## Key Metrics

**Phase 1**: Compare baseline (no first question) vs think-about-solution (best expected) vs other conditions.

**Phase 2**: Compare baseline (no numbers) vs transplant (with AI-generated numbers). Effect size = transplant - baseline.

## Data Integrity

Phase 2 only uses real AI-generated numbers from Phase 1. No fallback or synthetic data is used. Files are timestamped to track experiment sequence.

## Expected Results

**Phase 1**: Baseline should show lowest accuracy, "think about solution" should show highest accuracy.

**Phase 2**: Transplanted numbers should improve or maintain accuracy compared to baseline.

## Usage

Run `python main_phase1.py` to generate Phase 1 data, then `python main_phase2.py` to generate Phase 2 data. Phase 2 automatically finds and uses the latest Phase 1 results.
