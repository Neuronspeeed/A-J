# Analysis Tools and Reports

This directory contains analysis tools and reports for the Phase 2 thinking transplant experiment.

## Files Overview

### Reports
- **`PHASE2_RESULTS_REPORT.md`** - Professional markdown report with complete experimental results
- **`ANALYSIS_USAGE.md`** - Usage guide for analysis tools and interpretation

### Analysis Scripts
- **`generate_phase2_reports.py`** - Comprehensive report generator with verification
- **`analyze_phase2_results.py`** - Original analysis script with visualization support

## Quick Start

### Generate Summary Report
```bash
cd analysis
python generate_phase2_reports.py --summary-only
```

### Quick Verification
```bash
cd analysis
python -c "from generate_phase2_reports import quick_verification; quick_verification()"
```

### Full Analysis Report
```bash
cd analysis
python generate_phase2_reports.py --output detailed_report.txt
```

## Key Results Summary

### Overall Performance
- **Baseline**: 4.451 ± 10.057 digits correct
- **Transplanted**: 4.500 ± 10.233 digits correct (+1.1%)
- **Random**: 4.436 ± 9.491 digits correct (-0.3%)

### Top Performing Models
- **gpt-4.1-mini**: +45.3% improvement with transplanted numbers
- **gpt-4o-mini**: +61.3% improvement with transplanted numbers

### Data Quality
- **96.2%** completion rate (606/630 trials)
- **98.6%** valid XML responses
- **9** failed trials due to missing number sets

## Scientific Conclusion

The thinking transplant hypothesis is **partially validated**:
- Mechanism works for specific model architectures
- Smaller models show significant positive effects
- Larger models show minimal or negative effects
- Overall effect is measurable but model-dependent

## File Dependencies

The analysis scripts automatically locate Phase 2 data files in `../data/phase2/` directory. No manual file path specification required for standard usage.

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib (optional, for visualizations)
- seaborn (optional, for visualizations)

## Usage Examples

See `ANALYSIS_USAGE.md` for comprehensive usage examples and programmatic access patterns.
