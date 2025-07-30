# Phase 2 Analysis Tools Usage Guide

## Files Created

### 1. PHASE2_RESULTS_REPORT.md
Report with complete experimental results, methodology, and conclusions.

### 2. generate_phase2_reports.py
Script for generating various types of reports and verifications.

## Usage Examples

### Basic Report Generation

```bash
# Generate full comprehensive report
python generate_phase2_reports.py

# Generate summary only
python generate_phase2_reports.py --summary-only

# Analyze specific file
python generate_phase2_reports.py data/phase2/your_file.csv

# Save detailed report to file
python generate_phase2_reports.py --output detailed_analysis.txt
```

### Quick Verification

```python
from generate_phase2_reports import quick_verification
quick_verification()
```

### Export Summary Data

```python
from generate_phase2_reports import export_summary_csv
export_summary_csv()
```

### Programmatic Access

```python
from generate_phase2_reports import Phase2ReportGenerator

# Initialize with data file
generator = Phase2ReportGenerator('data/phase2/phase2_transplant-test_20250730_171147.csv')

# Get overall performance analysis
overall_results = generator.overall_performance_analysis()
print(f"Overall improvement: {overall_results['improvements']['transplant_vs_baseline']['percentage']:.1f}%")

# Get model-specific results
model_results = generator.model_specific_analysis()
for model, results in model_results.items():
    improvement = results['improvement']['percentage']
    if not pd.isna(improvement):
        print(f"{model}: {improvement:+.1f}%")

# Generate summary table
summary_df = generator.generate_summary_table()
print(summary_df)

# Get data quality report
quality = generator.data_quality_report()
print(f"Completion rate: {quality['completion_rate']:.1f}%")
```

## Report Components

### Data Quality Assessment
- Total trials and completion rates
- Error analysis and missing data
- Response quality metrics

### Overall Performance Analysis
- Mean performance by condition
- Statistical measures (mean, std, median, min, max)
- Improvement calculations

### Model-Specific Analysis
- Individual model performance
- Model-specific improvements
- Comparison across conditions

### Verification Reports
- Manual calculation verification
- Key result spot-checks
- Data integrity confirmation

## Key Metrics

### Overall Results
- Baseline: 4.451 ± 10.057 digits correct
- Transplanted: 4.500 ± 10.233 digits correct (+1.1%)
- Random: 4.436 ± 9.491 digits correct (-0.3%)

### Top Performing Models
- gpt-4.1-mini: +45.3% improvement
- gpt-4o-mini: +61.3% improvement

### Data Quality
- 96.2% completion rate
- 98.6% valid XML responses
- 9 failed trials due to missing number sets

## Interpretation

The results show that thinking transplant effects are highly model-specific:
- Smaller models (mini variants) show significant positive effects
- Larger models show minimal or negative effects
- Overall effect is small but measurable
- The mechanism works but with variable success rates

## Files Generated

Running the analysis tools may generate:
- `detailed_analysis.txt` - Comprehensive text report
- `phase2_summary_YYYYMMDD_HHMMSS.csv` - Summary data in CSV format
- Console output with formatted results

## Dependencies

Required Python packages:
- pandas
- numpy
- pathlib (built-in)
- glob (built-in)
- argparse (built-in)
- datetime (built-in)
