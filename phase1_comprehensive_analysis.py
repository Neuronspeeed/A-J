#!/usr/bin/env python3
"""
Comprehensive Phase 1 Analysis Script

Creates three separate analysis tables:
1. Problem-Level Analysis
2. Condition Analysis  
3. Model-Condition Analysis

Each analysis includes mean, count, and standard deviation of digits_correct.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def load_and_clean_data(csv_file_path):
    """Load data and handle missing values."""
    df = pd.read_csv(csv_file_path)
    
    # Convert digits_correct to numeric, handling any potential issues
    df['digits_correct_num'] = pd.to_numeric(df['digits_correct'], errors='coerce')
    
    # Remove rows with missing digits_correct values
    df_clean = df.dropna(subset=['digits_correct_num'])
    
    print(f"Data loaded: {len(df)} total trials, {len(df_clean)} valid trials")
    print(f"Missing data rate: {((len(df) - len(df_clean)) / len(df) * 100):.1f}%")
    
    return df_clean

def problem_level_analysis(df):
    """Analysis 1: Group by problem_id and problem_question."""
    print("\n" + "="*80)
    print("ANALYSIS 1: PROBLEM-LEVEL ANALYSIS")
    print("="*80)
    
    # Group by problem_id and problem_question
    problem_stats = df.groupby(['problem_id', 'problem_question'])['digits_correct_num'].agg([
        'mean', 'count', 'std'
    ]).reset_index()
    
    # Round for readability
    problem_stats['mean'] = problem_stats['mean'].round(3)
    problem_stats['std'] = problem_stats['std'].round(3)
    
    # Sort by mean performance (highest to lowest)
    problem_stats = problem_stats.sort_values('mean', ascending=False)
    
    # Display results
    print(f"{'Problem ID':<20} {'Mean':<8} {'Count':<6} {'Std':<8} {'Problem Question'}")
    print("-" * 120)
    
    for _, row in problem_stats.iterrows():
        question_short = row['problem_question'][:60] + "..." if len(row['problem_question']) > 60 else row['problem_question']
        print(f"{row['problem_id']:<20} {row['mean']:<8.3f} {row['count']:<6} {row['std']:<8.3f} {question_short}")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"Best performing problem: {problem_stats.iloc[0]['problem_id']} ({problem_stats.iloc[0]['mean']:.3f} digits)")
    print(f"Worst performing problem: {problem_stats.iloc[-1]['problem_id']} ({problem_stats.iloc[-1]['mean']:.3f} digits)")
    print(f"Performance range: {problem_stats['mean'].max() - problem_stats['mean'].min():.3f} digits")
    
    # Save to CSV
    problem_stats.to_csv('problem_level_analysis.csv', index=False)
    print("Results saved to: problem_level_analysis.csv")
    
    return problem_stats

def condition_analysis(df):
    """Analysis 2: Group by condition."""
    print("\n" + "="*80)
    print("ANALYSIS 2: CONDITION ANALYSIS")
    print("="*80)
    
    # Group by condition
    condition_stats = df.groupby('condition')['digits_correct_num'].agg([
        'mean', 'count', 'std'
    ]).reset_index()
    
    # Round for readability
    condition_stats['mean'] = condition_stats['mean'].round(3)
    condition_stats['std'] = condition_stats['std'].round(3)
    
    # Sort by mean performance (highest to lowest)
    condition_stats = condition_stats.sort_values('mean', ascending=False)
    
    # Display results
    print(f"{'Condition':<25} {'Mean':<8} {'Count':<6} {'Std':<8}")
    print("-" * 50)
    
    for _, row in condition_stats.iterrows():
        print(f"{row['condition']:<25} {row['mean']:<8.3f} {row['count']:<6} {row['std']:<8.3f}")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"Best condition: {condition_stats.iloc[0]['condition']} ({condition_stats.iloc[0]['mean']:.3f} digits)")
    print(f"Worst condition: {condition_stats.iloc[-1]['condition']} ({condition_stats.iloc[-1]['mean']:.3f} digits)")
    print(f"Performance range: {condition_stats['mean'].max() - condition_stats['mean'].min():.3f} digits")
    
    # Save to CSV
    condition_stats.to_csv('condition_analysis.csv', index=False)
    print("Results saved to: condition_analysis.csv")
    
    return condition_stats

def model_condition_analysis(df):
    """Analysis 3: Group by model and condition."""
    print("\n" + "="*80)
    print("ANALYSIS 3: MODEL-CONDITION ANALYSIS")
    print("="*80)
    
    # Group by model_name and condition
    model_condition_stats = df.groupby(['model_name', 'condition'])['digits_correct_num'].agg([
        'mean', 'count', 'std'
    ]).reset_index()
    
    # Round for readability
    model_condition_stats['mean'] = model_condition_stats['mean'].round(3)
    model_condition_stats['std'] = model_condition_stats['std'].round(3)
    
    # Sort by mean performance (highest to lowest)
    model_condition_stats = model_condition_stats.sort_values('mean', ascending=False)
    
    # Display results
    print(f"{'Model':<25} {'Condition':<20} {'Mean':<8} {'Count':<6} {'Std':<8}")
    print("-" * 70)
    
    for _, row in model_condition_stats.iterrows():
        model_short = row['model_name'].replace('claude-', '').replace('-20250514', '')
        print(f"{model_short:<25} {row['condition']:<20} {row['mean']:<8.3f} {row['count']:<6} {row['std']:<8.3f}")
    
    # Summary statistics
    best_combo = model_condition_stats.iloc[0]
    worst_combo = model_condition_stats.iloc[-1]
    
    print(f"\nSummary:")
    print(f"Best combination: {best_combo['model_name'].replace('claude-', '').replace('-20250514', '')} + {best_combo['condition']} ({best_combo['mean']:.3f} digits)")
    print(f"Worst combination: {worst_combo['model_name'].replace('claude-', '').replace('-20250514', '')} + {worst_combo['condition']} ({worst_combo['mean']:.3f} digits)")
    print(f"Performance range: {model_condition_stats['mean'].max() - model_condition_stats['mean'].min():.3f} digits")
    
    # Model averages
    model_averages = df.groupby('model_name')['digits_correct_num'].mean()
    print(f"\nModel Averages:")
    for model, avg in model_averages.items():
        model_short = model.replace('claude-', '').replace('-20250514', '')
        print(f"  {model_short}: {avg:.3f} digits")
    
    # Save to CSV
    model_condition_stats.to_csv('model_condition_analysis.csv', index=False)
    print("Results saved to: model_condition_analysis.csv")
    
    return model_condition_stats

def main():
    """Main analysis function."""
    # Default file path
    csv_file = 'data/phase1/phase1_thinking-experiment_20250805_122255.csv'
    
    # Allow command line argument for different file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    print(f"Analyzing: {csv_file}")
    
    try:
        # Load and clean data
        df = load_and_clean_data(csv_file)
        
        # Run all three analyses
        problem_stats = problem_level_analysis(df)
        condition_stats = condition_analysis(df)
        model_condition_stats = model_condition_analysis(df)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("Files generated:")
        print("- problem_level_analysis.csv")
        print("- condition_analysis.csv") 
        print("- model_condition_analysis.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
