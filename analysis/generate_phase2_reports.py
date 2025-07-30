#!/usr/bin/env python3
"""
Phase 2 Results Report Generator

This script generates  reports and verification analyses for the
Phase 2 thinking transplant experiment results.

Usage:
    python generate_phase2_reports.py [csv_file_path]
    
If no file path is provided, it will automatically find the latest Phase 2 results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import argparse
from datetime import datetime
from typing import Optional, Dict, Any


class Phase2ReportGenerator:
    """Generate comprehensive reports for Phase 2 experiment results."""
    
    def __init__(self, csv_file_path: str):
        """Initialize with the path to the Phase 2 results CSV file."""
        self.csv_file_path = csv_file_path
        self.df = None
        self.df_clean = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load and validate the Phase 2 results data."""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            # Remove rows with missing digits_correct for analysis
            self.df_clean = self.df.dropna(subset=['digits_correct'])
            print(f"Loaded {len(self.df)} total trials, {len(self.df_clean)} with valid results")
        except Exception as e:
            raise ValueError(f"Error loading data from {self.csv_file_path}: {e}")
    
    def data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality assessment."""
        report = {
            'total_trials': len(self.df),
            'valid_trials': len(self.df_clean),
            'missing_digits_correct': len(self.df) - len(self.df_clean),
            'error_trials': self.df['error'].notna().sum(),
            'xml_responses': self.df['full_response'].str.contains('<answer2>', na=False).sum(),
            'valid_math_answers': self.df['math_answer'].notna().sum(),
            'completion_rate': len(self.df_clean) / len(self.df) * 100,
        }
        
        # Error analysis
        if report['error_trials'] > 0:
            error_summary = self.df[self.df['error'].notna()]['error'].value_counts()
            report['error_types'] = error_summary.to_dict()
        
        # Model and condition distribution
        report['model_distribution'] = self.df['model_name'].value_counts().to_dict()
        report['condition_distribution'] = self.df['condition'].value_counts().to_dict()
        
        return report
    
    def overall_performance_analysis(self) -> Dict[str, Any]:
        """Analyze overall performance across conditions."""
        results = {}
        
        for condition in ['baseline_no_numbers', 'with_transplanted_numbers', 'with_random_numbers']:
            condition_data = self.df_clean[self.df_clean['condition'] == condition]['digits_correct']
            results[condition] = {
                'mean': condition_data.mean(),
                'std': condition_data.std(),
                'count': len(condition_data),
                'median': condition_data.median(),
                'min': condition_data.min(),
                'max': condition_data.max()
            }
        
        # Calculate improvements
        baseline_mean = results['baseline_no_numbers']['mean']
        transplant_mean = results['with_transplanted_numbers']['mean']
        random_mean = results['with_random_numbers']['mean']
        
        results['improvements'] = {
            'transplant_vs_baseline': {
                'absolute': transplant_mean - baseline_mean,
                'percentage': ((transplant_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
            },
            'random_vs_baseline': {
                'absolute': random_mean - baseline_mean,
                'percentage': ((random_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
            }
        }
        
        return results
    
    def model_specific_analysis(self) -> Dict[str, Any]:
        """Analyze performance by individual model."""
        results = {}
        
        for model in self.df_clean['model_name'].unique():
            model_data = self.df_clean[self.df_clean['model_name'] == model]
            model_results = {}
            
            for condition in ['baseline_no_numbers', 'with_transplanted_numbers', 'with_random_numbers']:
                condition_data = model_data[model_data['condition'] == condition]['digits_correct']
                model_results[condition] = {
                    'mean': condition_data.mean() if len(condition_data) > 0 else np.nan,
                    'std': condition_data.std() if len(condition_data) > 0 else np.nan,
                    'count': len(condition_data)
                }
            
            # Calculate improvement
            baseline = model_results['baseline_no_numbers']['mean']
            transplant = model_results['with_transplanted_numbers']['mean']
            
            if not pd.isna(baseline) and not pd.isna(transplant) and baseline > 0:
                improvement = transplant - baseline
                percentage = (improvement / baseline) * 100
            else:
                improvement = np.nan
                percentage = np.nan
            
            model_results['improvement'] = {
                'absolute': improvement,
                'percentage': percentage
            }
            
            results[model] = model_results
        
        return results
    
    def verification_report(self) -> Dict[str, Any]:
        """Generate verification report to confirm data integrity."""
        report = {}
        
        # Verify key calculations manually
        baseline_data = self.df_clean[self.df_clean['condition'] == 'baseline_no_numbers']['digits_correct']
        transplant_data = self.df_clean[self.df_clean['condition'] == 'with_transplanted_numbers']['digits_correct']
        
        report['manual_verification'] = {
            'baseline_mean': baseline_data.mean(),
            'transplant_mean': transplant_data.mean(),
            'improvement': transplant_data.mean() - baseline_data.mean(),
            'baseline_count': len(baseline_data),
            'transplant_count': len(transplant_data)
        }
        
        # Verify specific model results
        report['model_verification'] = {}
        for model in ['gpt-4.1-mini', 'gpt-4o-mini']:  # Key models with large effects
            model_data = self.df_clean[self.df_clean['model_name'] == model]
            baseline = model_data[model_data['condition'] == 'baseline_no_numbers']['digits_correct'].mean()
            transplant = model_data[model_data['condition'] == 'with_transplanted_numbers']['digits_correct'].mean()
            
            report['model_verification'][model] = {
                'baseline': baseline,
                'transplant': transplant,
                'improvement': transplant - baseline,
                'percentage': ((transplant - baseline) / baseline * 100) if baseline > 0 else 0
            }
        
        return report
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate a summary table of results by model and condition."""
        summary_data = []
        
        for model in self.df_clean['model_name'].unique():
            model_data = self.df_clean[self.df_clean['model_name'] == model]
            row = {'model_name': model}
            
            for condition in ['baseline_no_numbers', 'with_transplanted_numbers', 'with_random_numbers']:
                condition_data = model_data[model_data['condition'] == condition]['digits_correct']
                row[condition] = condition_data.mean() if len(condition_data) > 0 else np.nan
            
            # Calculate improvement
            baseline = row['baseline_no_numbers']
            transplant = row['with_transplanted_numbers']
            if not pd.isna(baseline) and not pd.isna(transplant):
                row['improvement'] = transplant - baseline
                row['percentage_change'] = ((transplant - baseline) / baseline * 100) if baseline > 0 else 0
            else:
                row['improvement'] = np.nan
                row['percentage_change'] = np.nan
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def print_comprehensive_report(self) -> None:
        """Print a comprehensive report to console."""
        print("=" * 80)
        print("PHASE 2 THINKING TRANSPLANT EXPERIMENT - COMPREHENSIVE REPORT")
        print("=" * 80)
        
        # Data Quality
        quality = self.data_quality_report()
        print(f"\nDATA QUALITY ASSESSMENT:")
        print(f"Total trials: {quality['total_trials']}")
        print(f"Valid trials: {quality['valid_trials']}")
        print(f"Completion rate: {quality['completion_rate']:.1f}%")
        print(f"Error trials: {quality['error_trials']}")
        
        # Overall Performance
        overall = self.overall_performance_analysis()
        print(f"\nOVERALL PERFORMANCE:")
        for condition, stats in overall.items():
            if condition != 'improvements':
                print(f"{condition:25s}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
        
        print(f"\nIMPROVEMENTS OVER BASELINE:")
        improvements = overall['improvements']
        print(f"Transplanted numbers: {improvements['transplant_vs_baseline']['absolute']:+.3f} digits ({improvements['transplant_vs_baseline']['percentage']:+.1f}%)")
        print(f"Random numbers:       {improvements['random_vs_baseline']['absolute']:+.3f} digits ({improvements['random_vs_baseline']['percentage']:+.1f}%)")
        
        # Model-Specific Results
        print(f"\nMODEL-SPECIFIC RESULTS:")
        model_results = self.model_specific_analysis()
        for model, results in model_results.items():
            improvement = results['improvement']
            if not pd.isna(improvement['absolute']):
                print(f"{model:25s}: {improvement['absolute']:+.3f} digits ({improvement['percentage']:+.1f}%)")
        
        # Summary Table
        print(f"\nSUMMARY TABLE:")
        summary_df = self.generate_summary_table()
        print(summary_df.round(3).to_string(index=False))
        
        # Verification
        verification = self.verification_report()
        print(f"\nVERIFICATION:")
        manual = verification['manual_verification']
        print(f"Manual calculation - Baseline: {manual['baseline_mean']:.3f}, Transplant: {manual['transplant_mean']:.3f}")
        print(f"Manual calculation - Improvement: {manual['improvement']:+.3f} digits")
        
        print("=" * 80)
    
    def save_detailed_report(self, output_file: str) -> None:
        """Save detailed analysis to a text file."""
        with open(output_file, 'w') as f:
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            self.print_comprehensive_report()
            
            # Additional detailed analysis
            print(f"\nDETAILED DATA QUALITY REPORT:")
            quality = self.data_quality_report()
            for key, value in quality.items():
                print(f"{key}: {value}")
            
            print(f"\nDETAILED MODEL ANALYSIS:")
            model_results = self.model_specific_analysis()
            for model, results in model_results.items():
                print(f"\n{model}:")
                for condition, stats in results.items():
                    if condition != 'improvement':
                        print(f"  {condition}: {stats}")
            
            # Restore stdout
            sys.stdout = original_stdout
        
        print(f"Detailed report saved to: {output_file}")


def find_latest_phase2_file() -> Optional[str]:
    """Find the most recent Phase 2 results file."""
    pattern = "data/phase2/phase2_transplant-test_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time, newest first
    latest_file = max(files, key=lambda f: Path(f).stat().st_mtime)
    return latest_file


def main():
    """Main function to run the report generator."""
    parser = argparse.ArgumentParser(description='Generate Phase 2 experiment reports')
    parser.add_argument('file', nargs='?', help='Path to Phase 2 results CSV file')
    parser.add_argument('--output', '-o', help='Output file for detailed report')
    parser.add_argument('--summary-only', action='store_true', help='Print only summary to console')
    
    args = parser.parse_args()
    
    # Find or use provided file
    if args.file:
        csv_file = args.file
    else:
        csv_file = find_latest_phase2_file()
        if not csv_file:
            print("No Phase 2 results files found!")
            return
    
    print(f"Analyzing: {csv_file}")
    
    # Generate reports
    generator = Phase2ReportGenerator(csv_file)
    
    if args.summary_only:
        # Print only key results
        overall = generator.overall_performance_analysis()
        improvements = overall['improvements']
        print(f"Overall improvement: {improvements['transplant_vs_baseline']['percentage']:+.1f}%")
        
        model_results = generator.model_specific_analysis()
        print("Top performers:")
        for model, results in model_results.items():
            pct = results['improvement']['percentage']
            if not pd.isna(pct) and pct > 10:
                print(f"  {model}: {pct:+.1f}%")
    else:
        # Print comprehensive report
        generator.print_comprehensive_report()
    
    # Save detailed report if requested
    if args.output:
        generator.save_detailed_report(args.output)


def quick_verification():
    """Quick verification function for spot-checking results."""
    csv_file = find_latest_phase2_file()
    if not csv_file:
        print("No Phase 2 results files found!")
        return

    generator = Phase2ReportGenerator(csv_file)
    verification = generator.verification_report()

    print("QUICK VERIFICATION RESULTS:")
    print("-" * 40)

    # Overall results
    manual = verification['manual_verification']
    print(f"Overall improvement: {manual['improvement']:+.3f} digits")
    print(f"Sample sizes: baseline={manual['baseline_count']}, transplant={manual['transplant_count']}")

    # Key model verification
    print("\nKey model verification:")
    for model, results in verification['model_verification'].items():
        print(f"{model}: {results['percentage']:+.1f}% improvement")

    # Data quality check
    quality = generator.data_quality_report()
    print(f"\nData quality: {quality['completion_rate']:.1f}% completion rate")
    print(f"Errors: {quality['error_trials']}/{quality['total_trials']} trials")


def export_summary_csv():
    """Export summary results to CSV for external analysis."""
    csv_file = find_latest_phase2_file()
    if not csv_file:
        print("No Phase 2 results files found!")
        return

    generator = Phase2ReportGenerator(csv_file)
    summary_df = generator.generate_summary_table()

    output_file = f"phase2_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"Summary exported to: {output_file}")


if __name__ == "__main__":
    main()
