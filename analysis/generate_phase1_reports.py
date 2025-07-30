#!/usr/bin/env python3
"""
Phase 1 Results Report Generator

Generates reports for Phase 1 thinking experiment results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import argparse
from typing import Optional, Dict, Any


class Phase1ReportGenerator:
    """Generate reports for Phase 1 experiment results."""
    
    def __init__(self, csv_file_path: str):
        """Initialize with the path to the Phase 1 results CSV file."""
        self.csv_file_path = csv_file_path
        self.df = None
        self.df_clean = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load and validate the Phase 1 results data."""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            # Convert digits_correct to numeric, handling any string concatenations
            self.df['digits_correct_num'] = pd.to_numeric(self.df['digits_correct'], errors='coerce')
            # Remove rows with missing digits_correct for analysis
            self.df_clean = self.df.dropna(subset=['digits_correct_num'])
            print(f"Loaded {len(self.df)} total trials, {len(self.df_clean)} with valid results")
        except Exception as e:
            raise ValueError(f"Error loading data from {self.csv_file_path}: {e}")
    
    def data_quality_report(self) -> Dict[str, Any]:
        """Generate data quality assessment."""
        report = {
            'total_trials': len(self.df),
            'valid_trials': len(self.df_clean),
            'missing_digits_correct': len(self.df) - len(self.df_clean),
            'completion_rate': len(self.df_clean) / len(self.df) * 100,
        }
        
        # Model and condition distribution
        report['model_distribution'] = self.df['model_name'].value_counts().to_dict()
        report['condition_distribution'] = self.df['condition'].value_counts().to_dict()
        
        return report
    
    def condition_analysis(self) -> Dict[str, Any]:
        """Analyze performance across conditions."""
        results = {}
        
        for condition in self.df_clean['condition'].unique():
            condition_data = self.df_clean[self.df_clean['condition'] == condition]['digits_correct_num']
            results[condition] = {
                'mean': condition_data.mean(),
                'std': condition_data.std(),
                'count': len(condition_data),
                'median': condition_data.median(),
                'min': condition_data.min(),
                'max': condition_data.max()
            }
        
        # Find best and worst conditions
        condition_means = {k: v['mean'] for k, v in results.items()}
        best_condition = max(condition_means, key=condition_means.get)
        worst_condition = min(condition_means, key=condition_means.get)
        
        results['summary'] = {
            'best_condition': best_condition,
            'best_mean': condition_means[best_condition],
            'worst_condition': worst_condition,
            'worst_mean': condition_means[worst_condition],
            'improvement': condition_means[best_condition] - condition_means[worst_condition]
        }
        
        return results
    
    def model_analysis(self) -> Dict[str, Any]:
        """Analyze performance by model."""
        results = {}
        
        for model in self.df_clean['model_name'].unique():
            model_data = self.df_clean[self.df_clean['model_name'] == model]
            model_results = {}
            
            for condition in model_data['condition'].unique():
                condition_data = model_data[model_data['condition'] == condition]['digits_correct_num']
                model_results[condition] = {
                    'mean': condition_data.mean() if len(condition_data) > 0 else np.nan,
                    'count': len(condition_data)
                }
            
            results[model] = model_results
        
        return results
    
    def hypothesis_test(self) -> Dict[str, Any]:
        """Test the core Phase 1 hypothesis."""
        baseline_data = self.df_clean[self.df_clean['condition'] == 'baseline']['digits_correct_num']
        think_data = self.df_clean[self.df_clean['condition'] == 'think_about_solution']['digits_correct_num']
        
        if len(baseline_data) == 0 or len(think_data) == 0:
            return {'error': 'Missing baseline or think_about_solution data'}
        
        baseline_mean = baseline_data.mean()
        think_mean = think_data.mean()
        improvement = think_mean - baseline_mean
        
        return {
            'baseline_mean': baseline_mean,
            'think_mean': think_mean,
            'improvement': improvement,
            'percentage_improvement': (improvement / baseline_mean * 100) if baseline_mean > 0 else 0,
            'baseline_count': len(baseline_data),
            'think_count': len(think_data)
        }
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table by condition."""
        summary_data = []
        
        for condition in self.df_clean['condition'].unique():
            condition_data = self.df_clean[self.df_clean['condition'] == condition]['digits_correct_num']
            summary_data.append({
                'condition': condition,
                'mean': condition_data.mean(),
                'std': condition_data.std(),
                'count': len(condition_data),
                'median': condition_data.median()
            })
        
        df_summary = pd.DataFrame(summary_data)
        return df_summary.sort_values('mean', ascending=False)
    
    def print_comprehensive_report(self) -> None:
        """Print comprehensive report to console."""
        print("=" * 80)
        print("PHASE 1 THINKING EXPERIMENT - COMPREHENSIVE REPORT")
        print("=" * 80)
        
        # Data Quality
        quality = self.data_quality_report()
        print(f"\nDATA QUALITY:")
        print(f"Total trials: {quality['total_trials']}")
        print(f"Valid trials: {quality['valid_trials']}")
        print(f"Completion rate: {quality['completion_rate']:.1f}%")
        
        # Hypothesis Test
        hypothesis = self.hypothesis_test()
        if 'error' not in hypothesis:
            print(f"\nHYPOTHESIS TEST:")
            print(f"Baseline: {hypothesis['baseline_mean']:.3f} digits correct")
            print(f"Think about solution: {hypothesis['think_mean']:.3f} digits correct")
            print(f"Improvement: {hypothesis['improvement']:+.3f} digits ({hypothesis['percentage_improvement']:+.1f}%)")
        
        # Condition Analysis
        conditions = self.condition_analysis()
        print(f"\nCONDITION ANALYSIS:")
        summary = conditions['summary']
        print(f"Best: {summary['best_condition']} ({summary['best_mean']:.3f})")
        print(f"Worst: {summary['worst_condition']} ({summary['worst_mean']:.3f})")
        print(f"Range: {summary['improvement']:.3f} digits")
        
        # Summary Table
        print(f"\nSUMMARY TABLE:")
        summary_df = self.generate_summary_table()
        print(summary_df.round(3).to_string(index=False))
        
        print("=" * 80)


def find_latest_phase1_file() -> Optional[str]:
    """Find the most recent Phase 1 results file."""
    pattern = "data/phase1/phase1_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    latest_file = max(files, key=lambda f: Path(f).stat().st_mtime)
    return latest_file


def main():
    """Main function to run the report generator."""
    parser = argparse.ArgumentParser(description='Generate Phase 1 experiment reports')
    parser.add_argument('file', nargs='?', help='Path to Phase 1 results CSV file')
    parser.add_argument('--summary-only', action='store_true', help='Print only summary')
    
    args = parser.parse_args()
    
    # Find or use provided file
    if args.file:
        csv_file = args.file
    else:
        csv_file = find_latest_phase1_file()
        if not csv_file:
            print("No Phase 1 results files found!")
            return
    
    print(f"Analyzing: {csv_file}")
    
    # Generate reports
    generator = Phase1ReportGenerator(csv_file)
    
    if args.summary_only:
        hypothesis = generator.hypothesis_test()
        if 'error' not in hypothesis:
            print(f"Thinking improvement: {hypothesis['percentage_improvement']:+.1f}%")
        
        conditions = generator.condition_analysis()
        summary = conditions['summary']
        print(f"Best condition: {summary['best_condition']} ({summary['best_mean']:.3f})")
    else:
        generator.print_comprehensive_report()


if __name__ == "__main__":
    main()
