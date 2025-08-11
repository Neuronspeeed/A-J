#!/usr/bin/env python3
"""
Analysis script to generate the requested table from phase1 data.
Creates a table with 3 columns: model, condition, avg_digits_correct
"""

import pandas as pd
import sys

def analyze_phase1_data(csv_file_path):
    """
    Analyze phase1 data and create the requested table.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Table with model, condition, avg_digits_correct
    """
    # Load the data
    df = pd.read_csv(csv_file_path)
    
    # Convert digits_correct to numeric, handling any potential issues
    df['digits_correct_num'] = pd.to_numeric(df['digits_correct'], errors='coerce')
    
    # Remove rows with missing digits_correct values
    df_clean = df.dropna(subset=['digits_correct_num'])
    
    # Group by model and condition, calculate mean digits correct
    result = df_clean.groupby(['model_name', 'condition'])['digits_correct_num'].mean().reset_index()
    result.columns = ['model', 'condition', 'avg_digits_correct']
    
    # Round to 3 decimal places for readability
    result['avg_digits_correct'] = result['avg_digits_correct'].round(3)
    
    # Sort by model first, then by avg_digits_correct descending within each model
    result = result.sort_values(['model', 'avg_digits_correct'], ascending=[True, False])
    
    return result

def print_formatted_table(result_df):
    """Print a nicely formatted table."""
    print('ANALYSIS RESULTS: Model, Condition, Average Digits Correct')
    print('=' * 65)
    print(f'{"Model":<25} {"Condition":<20} {"Avg Digits":<10}')
    print('-' * 65)
    
    for _, row in result_df.iterrows():
        print(f'{row["model"]:<25} {row["condition"]:<20} {row["avg_digits_correct"]:<10.3f}')
    
    print()
    print('SUMMARY INSIGHTS:')
    print('=' * 65)
    
    # Best performing combination
    best_row = result_df.loc[result_df['avg_digits_correct'].idxmax()]
    print(f'Best performance: {best_row["model"]} - {best_row["condition"]} ({best_row["avg_digits_correct"]:.3f} digits)')
    
    # Model averages
    model_averages = result_df.groupby('model')['avg_digits_correct'].mean()
    print('\nModel averages:')
    for model, avg in model_averages.items():
        model_short = model.replace('claude-', '').replace('-20250514', '')
        print(f'  {model_short}: {avg:.3f}')
    
    # Best condition per model
    print('\nBest condition per model:')
    for model in result_df['model'].unique():
        model_data = result_df[result_df['model'] == model]
        best_condition = model_data.loc[model_data['avg_digits_correct'].idxmax()]
        model_short = model.replace('claude-', '').replace('-20250514', '')
        print(f'  {model_short}: {best_condition["condition"]} ({best_condition["avg_digits_correct"]:.3f})')

def main():
    """Main function."""
    csv_file = 'data/phase1/phase1_thinking-experiment_20250805_122255.csv'
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    try:
        result_df = analyze_phase1_data(csv_file)
        print_formatted_table(result_df)
        
        # Save to CSV
        output_file = 'model_condition_analysis.csv'
        result_df.to_csv(output_file, index=False)
        print(f'\nTable saved to: {output_file}')
        
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
