"""
Data management utilities for the thinking transplant experiment.

This module handles file organization, naming conventions, and metadata
for experimental results.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .data_models import ExperimentPhase, ExperimentConfig, ExperimentResults


@dataclass
class DataPaths:
    """Centralized data path management."""
    
    # Base directories
    DATA_ROOT = Path("data")
    PHASE1_DIR = DATA_ROOT / "phase1"
    PHASE2_DIR = DATA_ROOT / "phase2" 
    PHASE3_DIR = DATA_ROOT / "phase3"
    ANALYSIS_DIR = DATA_ROOT / "analysis"
    
    @classmethod
    def ensure_directories_exist(cls):
        """Create all necessary directories if they don't exist."""
        for dir_path in [cls.DATA_ROOT, cls.PHASE1_DIR, cls.PHASE2_DIR, 
                        cls.PHASE3_DIR, cls.ANALYSIS_DIR]:
            dir_path.mkdir(exist_ok=True)


class DataManager:
    """
    Manages data storage, naming conventions, and metadata for experiments.
    
    Provides consistent file naming, metadata tracking, and easy data discovery.
    """
    
    def __init__(self):
        DataPaths.ensure_directories_exist()
    
    def generate_filename(
        self,
        phase: ExperimentPhase,
        experiment_name: Optional[str] = None,
        file_type: str = "csv",
        include_models: Optional[List[str]] = None
    ) -> str:
        """
        Generate a descriptive filename with timestamp and metadata.
        
        Args:
            phase: Experiment phase
            experiment_name: Optional custom name
            file_type: File extension (csv, json, etc.)
            include_models: List of models tested (for filename brevity)
            
        Returns:
            Formatted filename with full path
            
        Examples:
            phase1_thinking-experiment_6models_20250729_230045.csv
            phase2_transplant-test_gpt4-claude_20250729_230045.csv
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Base name from phase
        if phase == ExperimentPhase.PHASE_1:
            base_name = "phase1_thinking-experiment"
            output_dir = DataPaths.PHASE1_DIR
        elif phase == ExperimentPhase.PHASE_2:
            base_name = "phase2_transplant-test"
            output_dir = DataPaths.PHASE2_DIR
        elif phase == ExperimentPhase.PHASE_3:
            base_name = "phase3_cross-problem"
            output_dir = DataPaths.PHASE3_DIR
        else:
            base_name = "experiment"
            output_dir = DataPaths.DATA_ROOT
        
        # Add experiment name if provided
        if experiment_name:
            base_name += f"_{experiment_name.replace(' ', '-').lower()}"
        
        # Add model info for brevity
        if include_models:
            if len(include_models) <= 2:
                model_str = "-".join(m.replace("gpt-", "gpt").replace("claude-", "claude") 
                                   for m in include_models)
            else:
                model_str = f"{len(include_models)}models"
            base_name += f"_{model_str}"
        
        filename = f"{base_name}_{timestamp}.{file_type}"
        return str(output_dir / filename)
    
    def save_experiment_metadata(
        self, 
        results: ExperimentResults, 
        filename: str
    ) -> str:
        """
        Save experiment metadata alongside results for easy analysis.
        
        Args:
            results: Complete experiment results
            filename: Path to the main results file
            
        Returns:
            Path to the metadata file
        """
        metadata_file = filename.replace('.csv', '_metadata.json')
        
        metadata = {
            "experiment": {
                "name": results.config.name,
                "phase": results.config.phase.value,
                "description": results.config.description,
                "timestamp": results.start_time.isoformat() if results.start_time else None,
                "duration_seconds": results.total_duration_seconds
            },
            "configuration": {
                "conditions": [c.value for c in results.config.conditions],
                "models": results.config.model_names,
                "problems": [p.id for p in results.config.math_problems],
                "iterations_per_condition": results.config.iterations_per_condition
            },
            "results_summary": {
                "total_trials": results.total_trials,
                "successful_trials": results.successful_trials,
                "failed_trials": results.failed_trials,
                "success_rate": results.successful_trials / results.total_trials if results.total_trials > 0 else 0,
                "accuracy_by_condition": results.get_accuracy_by_condition()
            },
            "files": {
                "results_csv": filename,
                "metadata_json": metadata_file
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata_file
    
    def create_experiment_summary(self, results: ExperimentResults, filename: str) -> str:
        """
        Create a human-readable summary of the experiment.
        
        Args:
            results: Complete experiment results
            filename: Path to the main results file
            
        Returns:
            Path to the summary file
        """
        summary_file = filename.replace('.csv', '_summary.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("THINKING TRANSPLANT EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Experiment: {results.config.name}\n")
            f.write(f"Phase: {results.config.phase.value}\n")
            f.write(f"Date: {results.start_time.strftime('%Y-%m-%d %H:%M:%S') if results.start_time else 'Unknown'}\n")
            f.write(f"Duration: {results.total_duration_seconds:.1f} seconds\n\n")

            f.write("CONFIGURATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Models tested: {', '.join(results.config.model_names)}\n")
            f.write(f"Conditions: {', '.join(c.value for c in results.config.conditions)}\n")
            f.write(f"Math problems: {len(results.config.math_problems)}\n")
            f.write(f"Iterations per condition: {results.config.iterations_per_condition}\n\n")

            f.write("RESULTS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total trials: {results.total_trials}\n")
            f.write(f"Successful: {results.successful_trials}\n")
            f.write(f"Failed: {results.failed_trials}\n")
            f.write(f"Success rate: {100 * results.successful_trials / results.total_trials:.1f}%\n\n")

            accuracy_by_condition = results.get_accuracy_by_condition()
            if accuracy_by_condition:
                f.write("ACCURACY BY CONDITION (mean digits correct)\n")
                f.write("-" * 40 + "\n")
                for condition, accuracy in accuracy_by_condition.items():
                    f.write(f"{condition:25}: {accuracy:.2f}\n")
                f.write("\n")
            
            # Phase-specific analysis
            if results.config.phase == ExperimentPhase.PHASE_1:
                f.write("PHASE 1 HYPOTHESIS TEST\n")
                f.write("-" * 25 + "\n")
                baseline_acc = accuracy_by_condition.get('baseline', 0)
                think_acc = accuracy_by_condition.get('think_about_solution', 0)
                f.write(f"Baseline (expected worst): {baseline_acc:.2f}\n")
                f.write(f"Think about solution (expected best): {think_acc:.2f}\n")
                if think_acc > baseline_acc:
                    f.write("✅ HYPOTHESIS CONFIRMED: Thinking improves accuracy!\n")
                else:
                    f.write("❌ Hypothesis not confirmed\n")
                f.write("\n")

            elif results.config.phase == ExperimentPhase.PHASE_2:
                f.write("PHASE 2 TRANSPLANT ANALYSIS\n")
                f.write("-" * 30 + "\n")
                baseline_acc = accuracy_by_condition.get('baseline_no_numbers', 0)
                transplant_acc = accuracy_by_condition.get('with_transplanted_numbers', 0)
                difference = transplant_acc - baseline_acc
                f.write(f"Baseline (no numbers): {baseline_acc:.2f}\n")
                f.write(f"With transplanted numbers: {transplant_acc:.2f}\n")
                f.write(f"Difference: {difference:+.2f}\n")
                if transplant_acc > baseline_acc:
                    f.write("✅ TRANSPLANT SUCCESSFUL: Numbers improved performance!\n")
                elif transplant_acc < baseline_acc:
                    f.write("❌ TRANSPLANT FAILED: Numbers hurt performance\n")
                else:
                    f.write("➖ NO EFFECT: Numbers had no impact\n")
                f.write("\n")

            f.write("FILES\n")
            f.write("-" * 10 + "\n")
            f.write(f"Results: {filename}\n")
            f.write(f"Metadata: {filename.replace('.csv', '_metadata.json')}\n")
            f.write(f"Summary: {summary_file}\n")
        
        return summary_file
    
    def find_latest_results(self, phase: ExperimentPhase) -> Optional[str]:
        """
        Find the most recent results file for a given phase.

        Args:
            phase: Experiment phase to search for

        Returns:
            Path to the most recent results file, or None if not found
        """
        if phase == ExperimentPhase.PHASE_1:
            search_dir = DataPaths.PHASE1_DIR
            # Look for both old and new naming patterns
            patterns = ["phase1_*.csv", "phase1_thinking-experiment_*.csv"]
        elif phase == ExperimentPhase.PHASE_2:
            search_dir = DataPaths.PHASE2_DIR
            patterns = ["phase2_*.csv", "phase2_transplant-test_*.csv"]
        elif phase == ExperimentPhase.PHASE_3:
            search_dir = DataPaths.PHASE3_DIR
            patterns = ["phase3_*.csv", "phase3_cross-problem_*.csv"]
        else:
            return None

        import glob
        all_files = []

        # Search with all patterns
        for pattern in patterns:
            files = glob.glob(str(search_dir / pattern))
            all_files.extend(files)

        # NO LEGACY FALLBACK - this could mix old/new data and corrupt experiments

        if not all_files:
            return None

        # Return the most recently modified file
        return max(all_files, key=lambda f: Path(f).stat().st_mtime)
    
    def list_all_experiments(self) -> Dict[str, List[str]]:
        """
        List all experiment files organized by phase.
        
        Returns:
            Dictionary mapping phase names to lists of result files
        """
        experiments = {}
        
        for phase_name, phase_dir in [
            ("Phase 1", DataPaths.PHASE1_DIR),
            ("Phase 2", DataPaths.PHASE2_DIR), 
            ("Phase 3", DataPaths.PHASE3_DIR)
        ]:
            csv_files = list(phase_dir.glob("*.csv"))
            experiments[phase_name] = [str(f) for f in sorted(csv_files, reverse=True)]
        
        return experiments
