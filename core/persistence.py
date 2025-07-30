"""
Persistence layer for experiment results.

CSV persistence for experiment results.
"""

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol, TextIO
from datetime import datetime

from .data_models import TrialResult, ExperimentResults, MathProblem, ConditionType, ExperimentPhase


class ResultWriter(Protocol):
    """
    Protocol for writing experiment results.
    
    Enables dependency injection and easy testing.
    """
    
    def save_trial(self, trial: TrialResult) -> None:
        """Save a single trial result."""
        ...
    
    def save_experiment(self, results: ExperimentResults) -> None:
        """Save complete experiment results."""
        ...
    
    def finalize(self) -> str:
        """Finalize writing and return the output path."""
        ...


class ResultReader(Protocol):
    """
    Protocol for reading experiment results.
    
    Enables dependency injection and supports different storage formats.
    """
    
    def load_trials(self, filename: str) -> List[TrialResult]:
        """Load trial results from a file."""
        ...
    
    def load_experiment(self, filename: str) -> ExperimentResults:
        """Load complete experiment results from a file."""
        ...


class CsvResultWriter:
    """
    CSV implementation of ResultWriter.
    
    Writes results to CSV format for easy analysis in Excel, R, Python, etc.
    """
    
    def __init__(self, filename: str):
        self.filename = filename
        self.trials: List[TrialResult] = []
        self._file_handle: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None
        self._headers_written = False
    
    def save_trial(self, trial: TrialResult) -> None:
        """Save a single trial result to CSV."""
        self.trials.append(trial)
        
        # Initialize CSV writer on first trial
        if self._file_handle is None:
            self._file_handle = open(self.filename, 'w', newline='', encoding='utf-8')
            self._writer = csv.DictWriter(
                self._file_handle, 
                fieldnames=self._get_csv_headers()
            )
        
        # Write headers if not done yet
        if not self._headers_written and self._writer:
            self._writer.writeheader()
            self._headers_written = True

        # Convert trial to CSV row
        row = self._trial_to_csv_row(trial)
        if self._writer:
            self._writer.writerow(row)
        if self._file_handle:
            self._file_handle.flush()  # Ensure data is written immediately
    
    def save_experiment(self, results: ExperimentResults) -> None:
        """Save all trials from an experiment."""
        for trial in results.trials:
            self.save_trial(trial)
    
    def finalize(self) -> str:
        """Close file and return filename."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        return self.filename
    
    def _get_csv_headers(self) -> List[str]:
        """Define CSV column headers."""
        return [
            'trial_id', 'model_name', 'condition', 'phase',
            'problem_id', 'problem_question', 'expected_answer',
            'full_response', 'first_answer', 'math_answer',
            'generated_numbers', 'transplanted_numbers',
            'digits_correct', 'error', 'timestamp', 'duration_seconds'
        ]
    
    def _trial_to_csv_row(self, trial: TrialResult) -> Dict[str, Any]:
        """Convert a TrialResult to a CSV row."""
        return {
            'trial_id': trial.trial_id,
            'model_name': trial.model_name,
            'condition': trial.condition.value if hasattr(trial.condition, 'value') else trial.condition,
            'phase': trial.phase.value if hasattr(trial.phase, 'value') else trial.phase,
            'problem_id': trial.problem.id,
            'problem_question': trial.problem.question,
            'expected_answer': trial.problem.expected_answer,
            'full_response': trial.full_response,
            'first_answer': trial.first_answer,
            'math_answer': trial.math_answer,
            'generated_numbers': json.dumps(trial.generated_numbers) if trial.generated_numbers else None,
            'transplanted_numbers': json.dumps(trial.transplanted_numbers) if trial.transplanted_numbers else None,
            'digits_correct': trial.digits_correct,
            'error': trial.error,
            'timestamp': trial.timestamp.isoformat(),
            'duration_seconds': trial.duration_seconds
        }


class CsvResultReader:
    """
    CSV implementation of ResultReader.
    
    Reads results from CSV files created by CsvResultWriter.
    """
    
    def load_trials(self, filename: str) -> List[TrialResult]:
        """Load trial results from CSV file."""
        if not Path(filename).exists():
            raise FileNotFoundError(f"Results file not found: {filename}")
        
        trials = []
        
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                trial = self._csv_row_to_trial(row)
                trials.append(trial)
        
        return trials
    
    def load_experiment(self, filename: str) -> ExperimentResults:
        """
        REMOVED: This method creates fake experiment configs which is dangerous.

        Use load_trials() to get trial data only, or use metadata JSON files
        to get complete experiment information with proper configuration.
        """
        raise NotImplementedError(
            f"load_experiment() removed to prevent fake config generation. "
            f"Use load_trials('{filename}') to get trial data only, "
            f"or load the corresponding metadata JSON file for complete experiment data."
        )
    
    def _csv_row_to_trial(self, row: Dict[str, str]) -> TrialResult:
        """Convert a CSV row to a TrialResult."""
        # Parse JSON fields
        generated_numbers = None
        if row['generated_numbers']:
            try:
                generated_numbers = json.loads(row['generated_numbers'])
            except json.JSONDecodeError:
                pass
        
        transplanted_numbers = None
        if row['transplanted_numbers']:
            try:
                transplanted_numbers = json.loads(row['transplanted_numbers'])
            except json.JSONDecodeError:
                pass
        
        # Parse timestamp
        timestamp = datetime.fromisoformat(row['timestamp'])
        
        # Create MathProblem
        problem = MathProblem(
            id=row['problem_id'],
            question=row['problem_question'],
            expected_answer=row['expected_answer']
        )
        
        # Parse optional fields
        digits_correct = None
        if row['digits_correct']:
            try:
                digits_correct = int(row['digits_correct'])
            except ValueError:
                pass
        
        duration_seconds = None
        if row['duration_seconds']:
            try:
                duration_seconds = float(row['duration_seconds'])
            except ValueError:
                pass
        
        return TrialResult(
            trial_id=row['trial_id'],
            model_name=row['model_name'],
            condition=ConditionType(row['condition']),
            phase=ExperimentPhase(row['phase']),
            problem=problem,
            full_response=row['full_response'] or None,
            first_answer=row['first_answer'] or None,
            math_answer=row['math_answer'] or None,
            generated_numbers=generated_numbers,
            transplanted_numbers=transplanted_numbers,
            digits_correct=digits_correct,
            error=row['error'] or None,
            timestamp=timestamp,
            duration_seconds=duration_seconds
        )


class JsonResultWriter:
    """
    JSON implementation of ResultWriter.
    
    Alternative format that preserves more structure than CSV.
    """
    
    def __init__(self, filename: str):
        self.filename = filename
        self.trials: List[TrialResult] = []
    
    def save_trial(self, trial: TrialResult) -> None:
        """Save a single trial result."""
        self.trials.append(trial)
    
    def save_experiment(self, results: ExperimentResults) -> None:
        """Save complete experiment results."""
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(results.dict(), f, indent=2, default=str)
    
    def finalize(self) -> str:
        """
        REMOVED: This method creates fake experiment configs which is dangerous.

        Use save_experiment() with a real ExperimentResults object instead.
        """
        raise NotImplementedError(
            f"finalize() removed to prevent fake config generation. "
            f"Use save_experiment() with a real ExperimentResults object instead."
        )


def find_latest_results_file(pattern: str) -> Optional[str]:
    """
    Find the most recent results file matching a pattern.

    Used for Phase 2 to automatically find Phase 1 results.
    Returns None if no files found - no fallback behavior.

    Note: Consider using DataManager.find_latest_results() for better organization.
    """
    import glob

    files = glob.glob(pattern)
    if not files:
        return None

    # Return the most recently modified file
    return max(files, key=lambda f: Path(f).stat().st_mtime)


def find_all_results_files(pattern: str) -> List[str]:
    """
    Find all results files matching a pattern, sorted by modification time.
    
    Ensures that data is processed in a consistent, chronological order.
    """
    import glob
    
    files = glob.glob(pattern)
    if not files:
        return []
        
    # Sort files by modification time (oldest first)
    return sorted(files, key=lambda f: Path(f).stat().st_mtime)
