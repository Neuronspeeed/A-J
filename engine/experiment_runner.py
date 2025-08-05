"""
Core experiment engine for the thinking transplant study.
"""

import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Any
import time

from core.llm_providers import LLMProvider, LLMProviderError
from core.persistence import ResultWriter, CsvResultReader
from core.data_models import (
    ExperimentConfig, TrialResult, ExperimentResults, MathProblem,
    ConditionType, ExperimentPhase
)
from core.utils import (
    compare_decimal_strings, extract_xml_answers, extract_random_numbers,
    extract_numerical_answer, format_duration
)
from core.data_manager import DataManager
from config.experiments2 import get_prompt_template


class ExperimentRunner:
    """
    Core engine for running thinking transplant experiments.
    
    This class orchestrates the entire experimental process while remaining
    agnostic to specific LLM providers or storage mechanisms.
    """
    
    def __init__(self, provider: LLMProvider, writer: ResultWriter):
        """
        Initialize the experiment runner.

        Args:
            provider: LLM provider for making API calls
            writer: Result writer for saving data
        """
        self.provider = provider
        self.writer = writer
        self._harvested_numbers: Dict[str, Dict[str, List[List[int]]]] = {}
        self.data_manager = DataManager()
    
    async def run_experiment(self, config: ExperimentConfig) -> ExperimentResults:
        """
        Run a complete experiment according to the configuration.
        
        Args:
            config: Experiment configuration specifying what to test
            
        Returns:
            Complete experiment results with all trials
        """
        print(f"\n{'='*60}")
        print(f"Starting: {config.name}")
        print(f"Phase: {config.phase.value}")
        print(f"Description: {config.description.strip()}")
        print(f"{'='*60}")
        
        start_time = datetime.now()

        # Initialize results with explicit values - no defaults allowed
        results = ExperimentResults(
            config=config,
            total_trials=0,
            successful_trials=0,
            failed_trials=0,
            start_time=start_time,
            end_time=start_time,  # Will be updated at the end
            total_duration_seconds=0.0
        )
        
        # For Phase 2, load harvested numbers from Phase 1
        if config.phase == ExperimentPhase.PHASE_2:
            await self._load_harvested_numbers()
        
        total_trials = (
            len(config.math_problems) * 
            len(config.model_names) * 
            len(config.conditions) * 
            config.iterations_per_condition
        )
        
        print(f"Total trials planned: {total_trials}")
        trial_count = 0
        
        # Run all trials
        for problem in config.math_problems:
            for model_name in config.model_names:
                for condition in config.conditions:
                    for iteration in range(config.iterations_per_condition):
                        trial_count += 1
                        print(f"\nTrial {trial_count}/{total_trials}")
                        
                        trial = await self._run_single_trial(
                            problem=problem,
                            model_name=model_name,
                            condition=condition,
                            iteration=iteration,
                            config=config
                        )
                        
                        results.add_trial(trial)
                        self.writer.save_trial(trial)
        
        # Calculate final statistics explicitly - no defaults
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Count trials explicitly
        total_trials = len(results.trials)
        successful_trials = sum(1 for trial in results.trials if trial.error is None)
        failed_trials = total_trials - successful_trials

        # Update results with calculated values
        results.end_time = end_time
        results.total_duration_seconds = total_duration
        results.total_trials = total_trials
        results.successful_trials = successful_trials
        results.failed_trials = failed_trials
        
        print(f"\n{'='*60}")
        print(f"Experiment completed!")
        print(f"Total trials: {results.total_trials}")
        print(f"Successful: {results.successful_trials}")
        print(f"Failed: {results.failed_trials}")
        print(f"Duration: {results.total_duration_seconds:.1f} seconds")
        
        # Show accuracy summary
        accuracy_by_condition = results.get_accuracy_by_condition()
        if accuracy_by_condition:
            print(f"\nAccuracy by Condition (mean digits correct):")
            for condition, accuracy in accuracy_by_condition.items():
                print(f"  {condition}: {accuracy:.2f}")
        
        print(f"{'='*60}")
        
        return results
    
    async def _run_single_trial(
        self,
        problem: MathProblem,
        model_name: str,
        condition: ConditionType,
        iteration: int,
        config: ExperimentConfig
    ) -> TrialResult:
        """
        Run a single experimental trial.
        
        Args:
            problem: Math problem to solve
            model_name: Name of the model to test
            condition: Experimental condition
            iteration: Iteration number for this condition
            config: Experiment configuration
            
        Returns:
            Trial result with all data
        """
        trial_id = str(uuid.uuid4())
        start_time = time.time()
        
        print(f"  {model_name} | {condition.value} | {problem.id}")
        
        try:
            # Build prompts for this condition
            system_message, user_message, transplanted_numbers, random_numbers = self._build_prompts(
                problem, condition, iteration, model_name
            )
            
            # Get completion from LLM
            response = await self.provider.get_completion(
                system_message=system_message,
                user_message=user_message,
                timeout_seconds=config.timeout_seconds
            )
            
            # Parse response and calculate accuracy
            trial_data = self._parse_response(response, problem, condition)
            
            # Add transplanted numbers to trial data
            trial_data["transplanted_numbers"] = transplanted_numbers or random_numbers
            
            duration = time.time() - start_time
            
            trial = TrialResult(
                trial_id=trial_id,
                model_name=model_name,
                condition=condition,
                phase=config.phase,
                problem=problem,
                duration_seconds=duration,
                **trial_data
            )
            
            print(f"    Expected: {problem.expected_answer[:10]}...")
            print(f"    Got: {trial.math_answer}")
            print(f"    Digits correct: {trial.digits_correct}")
            
            return trial
            
        except Exception as e:
            duration = time.time() - start_time
            
            print(f"    ERROR: {str(e)}")
            
            return TrialResult(
                trial_id=trial_id,
                model_name=model_name,
                condition=condition,
                phase=config.phase,
                problem=problem,
                full_response="",  # No response due to error
                first_answer="",   # No answer due to error
                math_answer="",    # No answer due to error
                generated_numbers=[],  # No numbers due to error
                transplanted_numbers=[],  # No transplanted numbers due to error
                digits_correct=None,   # Cannot calculate due to error
                error=str(e),
                duration_seconds=duration
            )
    
    def _build_prompts(
        self,
        problem: MathProblem,
        condition: ConditionType,
        iteration: int,
        model_name: str
    ) -> tuple[str, str, list, list]:
        """
        Build system and user prompts for a specific condition.
        
        This preserves the exact prompt formats from the original specification.
        """
        prompt_template = get_prompt_template(condition)
        
        # Get transplanted numbers for Phase 2 if needed
        transplanted_numbers = None
        random_numbers = None

        if condition == ConditionType.WITH_TRANSPLANTED_NUMBERS:
            # Get numbers that were generated by the same model for the same problem
            problem_id = problem.id

            if (model_name in self._harvested_numbers and
                problem_id in self._harvested_numbers[model_name] and
                self._harvested_numbers[model_name][problem_id]):

                # Rotate through available numbers for this model+problem combination
                available_numbers = self._harvested_numbers[model_name][problem_id]
                numbers_index = iteration % len(available_numbers)
                transplanted_numbers = available_numbers[numbers_index]

            else:
                # NO FALLBACK - this is critical for experimental validity
                raise ValueError(
                    f"No harvested numbers available for model '{model_name}' "
                    f"and problem '{problem_id}'. Please ensure Phase 1 generated "
                    f"numbers for this specific model-problem combination."
                )

        elif condition == ConditionType.WITH_RANDOM_NUMBERS:
            # Generate completely random numbers as baseline (ONLY when explicitly requested)
            import random
            # Seed based on problem + iteration for reproducibility
            seed = hash(f"{problem.id}_{iteration}_{condition.value}") % (2**32)
            random.seed(seed)
            # Generate 3 random 10-digit numbers (same format as AI numbers)
            # This is NOT a fallback - it's a legitimate experimental condition
            random_numbers = [random.randint(1000000000, 9999999999) for _ in range(3)]
        
        # Build system message
        system_message = prompt_template["system"]
        if transplanted_numbers:
            system_message = system_message.format(
                transplanted_numbers=str(transplanted_numbers)
            )
        elif random_numbers:
            system_message = system_message.format(
                random_numbers=str(random_numbers)
            )
        
        # Build user message
        user_message = prompt_template["user_template"].format(
            math_question=problem.question
        )
        
        return system_message, user_message, transplanted_numbers, random_numbers
    
    def _parse_response(
        self, 
        response: str, 
        problem: MathProblem, 
        condition: ConditionType
    ) -> Dict[str, Any]:
        """
        Parse LLM response and extract relevant data.
        
        This implements the exact parsing logic from the original scripts.
        """
        if not response:
            return {
                "full_response": None,
                "first_answer": None,
                "math_answer": None,
                "generated_numbers": None,
                "transplanted_numbers": None,
                "digits_correct": None
            }
        
        # Extract answers based on condition type
        # All conditions now use XML format for reliable extraction
        if condition == ConditionType.BASELINE:
            # Baseline is math-only, no XML tags needed
            first_answer = None
            math_answer = extract_numerical_answer(response)
        elif condition in [
            ConditionType.THINK_ABOUT_SOLUTION, ConditionType.MEMORIZED,
            ConditionType.COMPLEX_STORY, ConditionType.PYTHON_PROGRAM,
            ConditionType.GENERATE_RANDOM_NUMBERS,
            ConditionType.BASELINE_NO_NUMBERS, ConditionType.WITH_TRANSPLANTED_NUMBERS,
            ConditionType.WITH_RANDOM_NUMBERS
        ]:
            # All other conditions use XML tags: Extract from XML tags
            first_answer, second_answer = extract_xml_answers(response)
            # Based on original experiment logic:
            # - Phase 2 conditions (math_first=True): math in first_answer
            # - Phase 1 non-baseline conditions (math_first=False): math in second_answer
            if condition in [ConditionType.BASELINE_NO_NUMBERS, 
                           ConditionType.WITH_TRANSPLANTED_NUMBERS, ConditionType.WITH_RANDOM_NUMBERS]:
                math_answer = first_answer  # Phase 2 conditions: math in first answer
            else:
                math_answer = second_answer  # Phase 1 non-baseline conditions: math in second answer
        else:
            # Fallback for any unknown conditions
            first_answer = None
            math_answer = extract_numerical_answer(response)

        # Extract generated numbers if this was the generate condition
        generated_numbers = None
        if condition == ConditionType.GENERATE_RANDOM_NUMBERS and first_answer:
            generated_numbers = extract_random_numbers(first_answer)
        
        # Calculate accuracy
        digits_correct = None
        if math_answer and problem.expected_answer:
            # All conditions now use XML format, so math_answer should be clean
            # But we still extract numerical answer to be safe
            numerical_answer = extract_numerical_answer(math_answer) if math_answer else None

            if numerical_answer:
                digits_correct = compare_decimal_strings(
                    problem.expected_answer, numerical_answer
                )
        
        return {
            "full_response": response,
            "first_answer": first_answer,
            "math_answer": math_answer,
            "generated_numbers": generated_numbers,
            "transplanted_numbers": None,  # Set by caller if needed
            "digits_correct": digits_correct
        }
    
    # Utility methods moved to core.utils for DRY principle
    
    async def _load_harvested_numbers(self) -> None:
        """Load harvested random numbers from Phase 1 results."""
        print("Loading harvested numbers from Phase 1...")

        # Find the most recent Phase 1 results using data manager
        phase1_file = self.data_manager.find_latest_results(ExperimentPhase.PHASE_1)
        if not phase1_file:
            print("  ❌ No Phase 1 results found!")
            print("  Please run Phase 1 first: uv run python main_phase1.py")
            # Don't set fallback numbers - this is scientifically invalid
            return
        
        print(f"  Loading from: {phase1_file}")
        
        # Load Phase 1 results using pandas (workaround for CSV parsing issue)
        import pandas as pd
        import json

        df = pd.read_csv(phase1_file)
        random_df = df[(df['condition'] == 'generate_random_numbers') & (df['generated_numbers'].notna())]

        print(f"  Found {len(random_df)} trials with generated numbers")
        
        # Extract numbers from generate_random_numbers condition, organized by model and problem
        for _, row in random_df.iterrows():
            model_name = row['model_name']
            problem_id = row['problem_id']
            numbers_str = row['generated_numbers']

            try:
                # Parse the JSON string to get the actual numbers
                numbers = json.loads(numbers_str)

                # Initialize nested dictionaries if needed
                if model_name not in self._harvested_numbers:
                    self._harvested_numbers[model_name] = {}
                if problem_id not in self._harvested_numbers[model_name]:
                    self._harvested_numbers[model_name][problem_id] = []

                # Add the numbers to the appropriate model+problem combination
                self._harvested_numbers[model_name][problem_id].append(numbers)

            except (json.JSONDecodeError, ValueError):
                # Skip invalid number strings
                continue
        
        if not self._harvested_numbers:
            print("  ❌ No AI-generated numbers found in Phase 1 results!")
            print("  Phase 2 transplant condition will fail without real AI numbers.")
            print("  Please ensure Phase 1 'generate_random_numbers' condition produced valid numbers.")
            # Don't set fallback - let the experiment fail properly
        else:
            # Count total number sets across all models and problems
            total_sets = sum(len(problems) for model_problems in self._harvested_numbers.values()
                           for problems in model_problems.values())
            print(f"  ✅ Harvested {total_sets} sets of AI-generated numbers")
            print(f"  Models: {list(self._harvested_numbers.keys())}")

            # Show example from first available model+problem
            first_model = next(iter(self._harvested_numbers.keys()))
            first_problem = next(iter(self._harvested_numbers[first_model].keys()))
            example_numbers = self._harvested_numbers[first_model][first_problem][0]
            print(f"  Example ({first_model}, {first_problem}): {example_numbers[:3]}...")
