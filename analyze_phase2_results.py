#!/usr/bin/env python3
"""
Phase 2 Results Analyzer - Thinking Transplant Experiment

This script analyzes the results of Phase 2 experiments to determine if the
"thinking transplant" hypothesis is supported by the data.

Usage:
    python analyze_phase2_results.py [path_to_csv_file]
    
If no file is provided, it will automatically find the latest Phase 2 results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import glob
from typing import Optional
import argparse


def find_latest_phase2_file() -> Optional[str]:
    """Find the most recent Phase 2 results file."""
    pattern = "data/phase2/phase2_transplant-test_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No Phase 2 results files found!")
        print(f"   Looking for pattern: {pattern}")
        return None
    
    # Sort by modification time, newest first
    latest_file = max(files, key=lambda f: Path(f).stat().st_mtime)
    return latest_file


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Load and validate the Phase 2 results data."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded data from: {file_path}")
        print(f"   Total trials: {len(df)}")
        
        # Validate required columns
        required_cols = ['model_name', 'condition', 'problem_id', 'digits_correct']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def analyze_overall_results(df: pd.DataFrame) -> dict:
    """Analyze overall experiment results."""
    print("\n" + "="*70)
    print("PHASE 2 THINKING TRANSPLANT EXPERIMENT - FINAL RESULTS")
    print("="*70)
    
    # Basic statistics
    total_trials = len(df)
    models_tested = df['model_name'].nunique()
    conditions_tested = df['condition'].nunique()
    problems_tested = df['problem_id'].nunique()
    
    print(f"📊 EXPERIMENT OVERVIEW:")
    print(f"   Total trials completed: {total_trials}")
    print(f"   Models tested: {models_tested}")
    print(f"   Conditions tested: {conditions_tested}")
    print(f"   Problems tested: {problems_tested}")
    
    # Check for completion
    expected_trials = models_tested * conditions_tested * problems_tested * 3  # 3 iterations per condition
    completion_rate = total_trials / expected_trials * 100 if expected_trials > 0 else 0
    print(f"   Completion rate: {completion_rate:.1f}%")
    
    # Condition balance
    print(f"\n📋 TRIALS BY CONDITION:")
    condition_counts = df['condition'].value_counts()
    for condition, count in condition_counts.items():
        print(f"   {condition}: {count} trials")
    
    return {
        'total_trials': total_trials,
        'models_tested': models_tested,
        'conditions_tested': conditions_tested,
        'problems_tested': problems_tested,
        'completion_rate': completion_rate
    }


def analyze_hypothesis_test(df: pd.DataFrame) -> dict:
    """Perform the core hypothesis test for thinking transplant."""
    print(f"\n🧪 CORE HYPOTHESIS TEST:")
    print(f"   H0: AI-generated numbers do NOT improve performance")
    print(f"   H1: AI-generated numbers DO improve performance")
    
    # Calculate mean accuracy by condition
    condition_stats = df.groupby('condition')['digits_correct'].agg(['mean', 'std', 'count', 'sem'])
    
    print(f"\n📈 ACCURACY BY CONDITION (mean digits correct):")
    for condition in condition_stats.index:
        mean_acc = condition_stats.loc[condition, 'mean']
        std_acc = condition_stats.loc[condition, 'std']
        count = condition_stats.loc[condition, 'count']
        sem = condition_stats.loc[condition, 'sem']
        print(f"   {condition:25s}: {mean_acc:.3f} ± {sem:.3f} (n={count})")
    
    # Extract key values
    baseline_mean = condition_stats.loc['baseline_no_numbers', 'mean']
    transplant_mean = condition_stats.loc['with_transplanted_numbers', 'mean']
    random_mean = condition_stats.loc['with_random_numbers', 'mean']
    
    # Calculate improvements
    transplant_improvement = transplant_mean - baseline_mean
    random_improvement = random_mean - baseline_mean
    
    transplant_pct = (transplant_improvement / baseline_mean * 100) if baseline_mean > 0 else 0
    random_pct = (random_improvement / baseline_mean * 100) if baseline_mean > 0 else 0
    
    print(f"\n🎯 HYPOTHESIS TEST RESULTS:")
    print(f"   Baseline (no numbers):        {baseline_mean:.3f} digits correct")
    print(f"   With transplanted numbers:    {transplant_mean:.3f} digits correct")
    print(f"   With random numbers:          {random_mean:.3f} digits correct")
    print(f"\n📊 IMPROVEMENTS OVER BASELINE:")
    print(f"   Transplanted numbers: {transplant_improvement:+.3f} digits ({transplant_pct:+.1f}%)")
    print(f"   Random numbers:       {random_improvement:+.3f} digits ({random_pct:+.1f}%)")
    
    # Determine hypothesis outcome
    print(f"\n🏆 CONCLUSION:")
    if transplant_mean > baseline_mean and transplant_improvement > 0.05:  # Meaningful improvement threshold
        if transplant_mean > random_mean:
            print(f"   🎉 HYPOTHESIS FULLY CONFIRMED!")
            print(f"      ✅ AI-generated numbers improve performance")
            print(f"      ✅ They work better than random numbers")
            conclusion = "FULLY_CONFIRMED"
        else:
            print(f"   ✅ HYPOTHESIS PARTIALLY CONFIRMED!")
            print(f"      ✅ AI-generated numbers improve performance")
            print(f"      ⚠️  But random numbers work equally well or better")
            conclusion = "PARTIALLY_CONFIRMED"
    else:
        print(f"   ❌ HYPOTHESIS NOT CONFIRMED")
        print(f"      ❌ No meaningful improvement from AI-generated numbers")
        conclusion = "NOT_CONFIRMED"
    
    return {
        'baseline_mean': baseline_mean,
        'transplant_mean': transplant_mean,
        'random_mean': random_mean,
        'transplant_improvement': transplant_improvement,
        'random_improvement': random_improvement,
        'transplant_pct': transplant_pct,
        'random_pct': random_pct,
        'conclusion': conclusion,
        'condition_stats': condition_stats
    }


def analyze_by_model(df: pd.DataFrame) -> dict:
    """Analyze results broken down by model."""
    print(f"\n🤖 RESULTS BY MODEL:")
    
    model_results = df.groupby(['model_name', 'condition'])['digits_correct'].mean().unstack()
    
    print(f"\n📊 Mean digits correct by model and condition:")
    print(model_results.round(3))
    
    # Calculate improvement for each model
    print(f"\n📈 Transplant improvement by model:")
    for model in model_results.index:
        baseline = model_results.loc[model, 'baseline_no_numbers']
        transplant = model_results.loc[model, 'with_transplanted_numbers']
        improvement = transplant - baseline
        pct_improvement = (improvement / baseline * 100) if baseline > 0 else 0
        print(f"   {model:25s}: {improvement:+.3f} digits ({pct_improvement:+.1f}%)")
    
    return {'model_results': model_results}


def create_visualizations(df: pd.DataFrame, output_dir: str = "data/analysis") -> None:
    """Create visualizations of the results."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall condition comparison
    plt.figure(figsize=(10, 6))
    condition_means = df.groupby('condition')['digits_correct'].agg(['mean', 'sem'])
    
    bars = plt.bar(range(len(condition_means)), condition_means['mean'], 
                   yerr=condition_means['sem'], capsize=5, alpha=0.8)
    plt.xlabel('Condition')
    plt.ylabel('Mean Digits Correct')
    plt.title('Phase 2: Thinking Transplant Results\nMean Accuracy by Condition')
    plt.xticks(range(len(condition_means)), condition_means.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, condition_means['mean'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phase2_condition_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Results by model
    plt.figure(figsize=(14, 8))
    model_pivot = df.pivot_table(values='digits_correct', index='model_name', 
                                columns='condition', aggfunc='mean')
    
    sns.heatmap(model_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Mean Digits Correct'})
    plt.title('Phase 2: Results by Model and Condition')
    plt.xlabel('Condition')
    plt.ylabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phase2_model_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualizations saved to: {output_dir}/")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze Phase 2 thinking transplant results')
    parser.add_argument('file', nargs='?', help='Path to Phase 2 results CSV file')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    args = parser.parse_args()
    
    # Find or use provided file
    if args.file:
        file_path = args.file
    else:
        file_path = find_latest_phase2_file()
        if not file_path:
            return
    
    # Load and analyze data
    df = load_and_validate_data(file_path)
    
    # Run analyses
    overview = analyze_overall_results(df)
    hypothesis_results = analyze_hypothesis_test(df)
    model_results = analyze_by_model(df)
    
    # Create visualizations
    if not args.no_plots:
        try:
            create_visualizations(df)
        except ImportError:
            print("\n⚠️  Matplotlib/Seaborn not available - skipping plots")
        except Exception as e:
            print(f"\n⚠️  Error creating plots: {e}")
    
    # Summary
    print(f"\n" + "="*70)
    print(f"🎉 ANALYSIS COMPLETE!")
    print(f"   Conclusion: {hypothesis_results['conclusion']}")
    if hypothesis_results['transplant_improvement'] > 0:
        print(f"   AI numbers improved performance by {hypothesis_results['transplant_pct']:.1f}%")
    print(f"="*70)


if __name__ == "__main__":
    main()
