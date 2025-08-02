# Analysis Tools and Reports

This directory contains analysis tools and reports for both Phase 1 and Phase 2 experiments.

## Files Overview

### Main Analysis Scripts
- **`comprehensive_verification.py`** - Complete verification with visualizations (recommended)
- **`generate_phase1_reports.py`** - Phase 1 analysis and report generator
- **`generate_phase2_reports.py`** - Phase 2 analysis and report generator

### Documentation
- **`PHASE2_RESULTS_REPORT.md`** - Professional markdown report with complete experimental results
- **`ANALYSIS_USAGE.md`** - Usage guide for analysis tools and interpretation

### Legacy Scripts
- **`analyze_phase2_results.py`** - Original Phase 2 analysis script

## Quick Start

### Complete Verification (Recommended)
```bash
python analysis/comprehensive_verification.py
```
Generates complete verification with all calculations, data quality checks, and visualizations saved to `data/` folder.

### Individual Analysis
```bash
python analysis/generate_phase1_reports.py --summary-only
python analysis/generate_phase1_reports.py --visualize  # With charts
python analysis/generate_phase2_reports.py --summary-only
python analysis/generate_phase2_reports.py --visualize  # With charts
```

### Full Reports
```bash
python analysis/generate_phase1_reports.py
python analysis/generate_phase2_reports.py
```

## Verified Results Summary

### Phase 1: Thinking While Distracted
- **1,346 valid trials** (98.4% completion rate)
- **Baseline**: 3.180 digits correct
- **Think about solution**: 3.973 digits correct (+24.9% improvement)
- **Best condition**: memorized (4.261 digits correct)

### Phase 2: Thinking Transplant
- **606 valid trials** (96.2% completion rate)
- **Overall improvement**: +1.1% with transplanted numbers
- **Top performers**: gpt-4o-mini (+61.3%), gpt-4.1-mini (+45.3%)
- **Interference effects**: gpt-4.1 (-50.0%), gpt-4.1-nano (-19.5%)

### Generated Visualizations
- `data/comprehensive_analysis.png` - Complete 4-panel overview
- `data/phase1_detailed_analysis.png` - Phase 1 detailed analysis
- `data/phase2_detailed_analysis.png` - Phase 2 detailed analysis
- `data/model_comparison.png` - Cross-phase model comparison

## Scientific Conclusion

**Phase 1**: Models do think about problems while working on unrelated tasks (+24.9% improvement)

**Phase 2**: Thinking transplant hypothesis is **partially validated**:
- Mechanism works for specific model architectures
- Smaller models show significant positive effects
- Larger models show interference effects
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
