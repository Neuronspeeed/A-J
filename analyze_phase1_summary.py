#!/usr/bin/env python3
"""
Phase 1 Data Analysis - Summary Table
Generates a summary table grouped by model and condition with average digits correct
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_phase1_data(csv_file_path: str):
    """Load and analyze Phase 1 experimental data."""
    
    # Load the CSV file
    print(f"Loading data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    # Convert digits_correct to numeric, handling any errors
    df['digits_correct_num'] = pd.to_numeric(df['digits_correct'], errors='coerce')
    
    # Remove rows with missing digits_correct for analysis
    df_clean = df.dropna(subset=['digits_correct_num'])
    
    print(f"Total trials loaded: {len(df)}")
    print(f"Valid trials with digits_correct: {len(df_clean)}")
    print()
    
    # Group by model_name and condition, calculate mean
    grouped = df_clean.groupby(['model_name', 'condition'])['digits_correct_num'].agg([
        ('count', 'count'),
        ('mean', 'mean')
    ]).reset_index()
    
    # Sort by model_name first, then by mean (descending) within each model
    grouped = grouped.sort_values(['model_name', 'mean'], ascending=[True, False])
    
    # Format the table
    print("=" * 80)
    print("PHASE 1 EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<30} {'Condition':<25} {'Avg Digits Correct':>20}")
    print("-" * 80)
    
    current_model = None
    for _, row in grouped.iterrows():
        if current_model != row['model_name']:
            if current_model is not None:
                print("-" * 80)
            current_model = row['model_name']
        
        print(f"{row['model_name']:<30} {row['condition']:<25} {row['mean']:>20.3f}")
    
    print("=" * 80)
    
    # Additional summary statistics
    print("\nOVERALL STATISTICS BY CONDITION (across all models):")
    print("-" * 60)
    overall_by_condition = df_clean.groupby('condition')['digits_correct_num'].agg([
        ('n', 'count'),
        ('mean', 'mean'),
        ('std', 'std')
    ]).sort_values('mean', ascending=False)
    
    print(f"{'Condition':<25} {'N':>8} {'Mean':>12} {'Std Dev':>12}")
    print("-" * 60)
    for condition, row in overall_by_condition.iterrows():
        print(f"{condition:<25} {row['n']:>8.0f} {row['mean']:>12.3f} {row['std']:>12.3f}")
    
    print("\nOVERALL STATISTICS BY MODEL:")
    print("-" * 60)
    overall_by_model = df_clean.groupby('model_name')['digits_correct_num'].agg([
        ('n', 'count'),
        ('mean', 'mean'),
        ('std', 'std')
    ]).sort_values('mean', ascending=False)
    
    print(f"{'Model':<30} {'N':>8} {'Mean':>12} {'Std Dev':>12}")
    print("-" * 60)
    for model, row in overall_by_model.iterrows():
        print(f"{model:<30} {row['n']:>8.0f} {row['mean']:>12.3f} {row['std']:>12.3f}")
    
    # Create a pivot table for better visualization
    print("\n\nPIVOT TABLE: Average Digits Correct")
    print("=" * 80)
    pivot = df_clean.pivot_table(
        values='digits_correct_num',
        index='condition',
        columns='model_name',
        aggfunc='mean'
    )
    
    # Sort conditions by overall mean performance
    condition_means = df_clean.groupby('condition')['digits_correct_num'].mean().sort_values(ascending=False)
    pivot = pivot.reindex(condition_means.index)
    
    # Format and display pivot table
    print(pivot.round(3).to_string())
    
    return grouped

if __name__ == "__main__":
    csv_path = "/Users/BrainTech/Documents/AI/A-J/data/phase1/phase1_thinking-experiment_20250805_122255.csv"
    
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Error: File not found at {csv_path}")
    else:
        results = analyze_phase1_data(csv_path)