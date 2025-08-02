#!/usr/bin/env python3
"""
Phase 1 Results Report Generator

Generates reports for Phase 1 thinking experiment results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    def generate_visualizations(self, save_path: str = "phase1_analysis.png") -> None:
        """Generate visualizations for Phase 1 results."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Phase 1: Thinking While Distracted - Analysis', fontsize=16, fontweight='bold')

        # 1. Performance by condition (bar plot)
        condition_means = self.df_clean.groupby('condition')['digits_correct_num'].mean().sort_values(ascending=False)
        condition_std = self.df_clean.groupby('condition')['digits_correct_num'].std()

        bars = axes[0,0].bar(range(len(condition_means)), condition_means.values,
                            yerr=condition_std[condition_means.index].values,
                            capsize=5, alpha=0.8, color='steelblue')
        axes[0,0].set_xticks(range(len(condition_means)))
        axes[0,0].set_xticklabels(condition_means.index, rotation=45, ha='right')
        axes[0,0].set_ylabel('Mean Digits Correct')
        axes[0,0].set_title('Performance by Condition')
        axes[0,0].grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, condition_means.values)):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                          f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        # 2. Distribution by condition (box plot)
        conditions_ordered = condition_means.index.tolist()
        condition_data = [self.df_clean[self.df_clean['condition'] == cond]['digits_correct_num'].values
                         for cond in conditions_ordered]

        bp = axes[0,1].boxplot(condition_data, labels=conditions_ordered, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        axes[0,1].set_xticklabels(conditions_ordered, rotation=45, ha='right')
        axes[0,1].set_ylabel('Digits Correct')
        axes[0,1].set_title('Performance Distribution by Condition')
        axes[0,1].grid(axis='y', alpha=0.3)

        # 3. Model performance comparison
        if 'model_name' in self.df_clean.columns:
            model_means = self.df_clean.groupby('model_name')['digits_correct_num'].mean().sort_values(ascending=True)
            y_pos = range(len(model_means))

            bars = axes[1,0].barh(y_pos, model_means.values, alpha=0.8, color='lightcoral')
            axes[1,0].set_yticks(y_pos)
            axes[1,0].set_yticklabels([name.replace('-', '-\n') for name in model_means.index], fontsize=9)
            axes[1,0].set_xlabel('Mean Digits Correct')
            axes[1,0].set_title('Performance by Model')
            axes[1,0].grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, model_means.values)):
                axes[1,0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                              f'{value:.2f}', ha='left', va='center', fontweight='bold')

        # 4. Improvement analysis
        baseline_mean = self.df_clean[self.df_clean['condition'] == 'baseline']['digits_correct_num'].mean()
        improvements = []
        condition_names = []

        for condition in condition_means.index:
            if condition != 'baseline':
                improvement = ((condition_means[condition] - baseline_mean) / baseline_mean) * 100
                improvements.append(improvement)
                condition_names.append(condition)

        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = axes[1,1].bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
        axes[1,1].set_xticks(range(len(improvements)))
        axes[1,1].set_xticklabels(condition_names, rotation=45, ha='right')
        axes[1,1].set_ylabel('Improvement over Baseline (%)')
        axes[1,1].set_title('Improvement Analysis')
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,1].grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, improvements)):
            axes[1,1].text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + (1 if value > 0 else -3),
                          f'{value:+.1f}%', ha='center', va='bottom' if value > 0 else 'top',
                          fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {save_path}")
        plt.show()


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
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
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

    # Generate visualizations if requested
    if args.visualize:
        generator.generate_visualizations()


if __name__ == "__main__":
    main()
