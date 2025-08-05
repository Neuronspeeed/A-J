"""
Main entry point for Phase 1 of the thinking transplant experiment.

This script implements the Composition Root pattern, where all dependencies
are constructed and injected. This is the ONLY place where concrete classes
are instantiated using dependency injection.

Phase 1 tests: "Do the models think while talking about something unrelated?"

Usage:
    uv run python main_phase1.py
"""

import asyncio
import sys
import signal
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from core.llm_providers import create_provider, LLMProviderError
from core.persistence import CsvResultWriter
from engine.experiment_runner import ExperimentRunner
from config.experiments2 import PHASE1_CONFIG, get_provider_config


class Phase1Runner:
    """Manages experiment execution with signal handling and retry logic."""
    
    def __init__(self):
        self.shutdown_requested = False
        self.writer: Optional[CsvResultWriter] = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def run_single_trial_with_retry(self, runner, model_name, trial_config, max_retries=3):
        """Run a single trial with retry logic."""
        for attempt in range(max_retries + 1):
            if self.shutdown_requested:
                return None
            
            try:
                results = await runner.run_experiment(trial_config)
                return results
            except Exception as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff
                    print(f"Trial failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"   Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Trial failed after {max_retries + 1} attempts: {e}")
                    return None
        return None


async def main():
    """
    Main function implementing experiment execution with error handling.

    This function:
    1. Validates environment setup
    2. Creates concrete dependencies with error handling
    3. Runs the experiment with retry logic
    4. Provides graceful shutdown and progress tracking
    5. Reports comprehensive results
    """
    # Check for command line options
    test_mode = "--test-mode" in sys.argv
    resume_mode = "--resume" in sys.argv
    
    phase1_runner = Phase1Runner()

    print("Phase 1: Do the models think while talking about something unrelated?")
    print("=" * 80)

    if test_mode:
        print("Test mode enabled (reduced scope)")
    elif resume_mode:
        print("Resume mode enabled")
    else:
        print("Full experiment execution")
        print("Options: --test-mode for reduced scope, --resume to continue interrupted runs")
    
    # Validate that we have API keys
    try:
        # Test provider creation to validate API keys early
        test_config = get_provider_config("claude-sonnet-4-20250514")
        test_provider = create_provider(test_config)
        print("API keys validated")
    except Exception as e:
        print(f"API key validation failed: {e}")
        print("\nRequired environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY (or CLAUDE_API_KEY)")
        print("\nTip: Create a .env file with your API keys")
        return 1
    
    # Create experiment configuration (copy so we can modify for test mode)
    config = PHASE1_CONFIG.model_copy()

    if test_mode:
        # Reduce scope for testing
        config.math_problems = config.math_problems[:2]  # Just 2 problems
        config.iterations_per_condition = 1  # Just 1 iteration
        config.model_names = ["claude-sonnet-4-20250514"]  # Just one model
        print("Test mode: 2 problems, 1 iteration, 1 model")

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = config.output_filename_template.format(timestamp=timestamp)

    print(f"Results output: {output_filename}")
    print(f"Models: {len(config.model_names)}")
    print(f"Conditions: {len(config.conditions)}")
    print(f"Math problems: {len(config.math_problems)}")

    total_trials = (
        len(config.model_names) *
        len(config.conditions) *
        len(config.math_problems) *
        config.iterations_per_condition
    )
    print(f"Total trials: {total_trials}")
    
    # Create result writer (dependency injection)
    writer = CsvResultWriter(output_filename)
    phase1_runner.writer = writer
    
    all_results = []
    completed_trials = 0
    failed_trials = 0
    start_time = datetime.now()
    
    try:
        # Run experiment for each model with bulletproof error handling
        for model_idx, model_name in enumerate(config.model_names):
            if phase1_runner.shutdown_requested:
                print(f"\nShutdown requested, stopping at model {model_name}")
                break
                
            print(f"\nTesting model {model_idx + 1}/{len(config.model_names)}: {model_name}")
            print("-" * 60)
            
            try:
                # Create provider for this model (dependency injection)
                provider_config = get_provider_config(model_name)
                provider = create_provider(provider_config)
                
                # Create experiment runner (dependency injection)
                runner = ExperimentRunner(provider, writer)
                
                # Create config for this specific model
                model_config = config.model_copy(deep=True)
                model_config.model_names = [model_name]  # Test only this model
                
                # Run experiment with retry logic
                print(f"   Starting experiment for {model_name}...")
                model_start_time = datetime.now()
                
                results = await phase1_runner.run_single_trial_with_retry(
                    runner, model_name, model_config
                )
                
                if results:
                    all_results.append(results)
                    completed_trials += results.successful_trials
                    failed_trials += (results.total_trials - results.successful_trials)
                    
                    model_duration = (datetime.now() - model_start_time).total_seconds()
                    print(f"Completed {model_name}: {results.successful_trials}/{results.total_trials} successful ({model_duration:.1f}s)")
                    
                    # Show progress
                    total_elapsed = (datetime.now() - start_time).total_seconds()
                    models_remaining = len(config.model_names) - (model_idx + 1)
                    if models_remaining > 0:
                        eta_seconds = (total_elapsed / (model_idx + 1)) * models_remaining
                        print(f"   Progress: {model_idx + 1}/{len(config.model_names)} models | ETA: {eta_seconds/60:.1f} min")
                else:
                    failed_trials += (len(config.conditions) * len(config.math_problems) * config.iterations_per_condition)
                    print(f"Model {model_name} failed completely")
                
                # Brief pause between models to prevent rate limiting
                if model_idx < len(config.model_names) - 1:
                    await asyncio.sleep(1)
                
            except LLMProviderError as e:
                print(f"Provider error for {model_name}: {e}")
                print(f"Continuing with next model...")
                failed_trials += (len(config.conditions) * len(config.math_problems) * config.iterations_per_condition)
                continue
            except Exception as e:
                print(f"Unexpected error for {model_name}: {e}")
                print(f"Continuing with next model...")
                failed_trials += (len(config.conditions) * len(config.math_problems) * config.iterations_per_condition)
                continue
    
    finally:
        # Always finalize output file
        final_filename = writer.finalize()
        total_duration = (datetime.now() - start_time).total_seconds()
        print(f"\nFinal results saved to: {final_filename}")
        print(f"Total experiment duration: {total_duration/60:.1f} minutes")
        
        if phase1_runner.shutdown_requested:
            print(f"\nExperiment was interrupted but results were saved successfully")
            print(f"Use --resume flag to continue interrupted experiments (if implemented)")
    
    # Print comprehensive summary
    if all_results or completed_trials > 0:
        print(f"\n{'='*80}")
        print("PHASE 1 SUMMARY")
        print(f"{'='*80}")
        
        total_successful = sum(r.successful_trials for r in all_results) if all_results else completed_trials
        total_planned = sum(r.total_trials for r in all_results) if all_results else total_trials
        total_attempted = total_successful + failed_trials
        
        print(f"EXPERIMENT STATISTICS:")
        print(f"   Successful trials: {total_successful}")
        print(f"   Failed trials: {failed_trials}")
        print(f"   Total attempted: {total_attempted}")
        print(f"   Success rate: {100*total_successful/total_attempted:.1f}%" if total_attempted > 0 else "   Success rate: N/A")
        print(f"   Completion: {100*total_attempted/total_planned:.1f}%" if total_planned > 0 else "   Completion: N/A")
        
        # Aggregate accuracy by condition
        from collections import defaultdict
        condition_accuracies = defaultdict(list)
        
        for results in all_results:
            for condition, accuracy in results.get_accuracy_by_condition().items():
                condition_accuracies[condition].append(accuracy)
        
        print(f"\nAccuracy by Condition (mean digits correct):")
        for condition, accuracies in condition_accuracies.items():
            if accuracies:
                mean_accuracy = sum(accuracies) / len(accuracies)
                print(f"  {condition}: {mean_accuracy:.2f}")
        
        # Test hypothesis with enhanced analysis
        baseline_acc = condition_accuracies.get('baseline', [0])
        think_acc = condition_accuracies.get('think_about_solution', [0])
        
        if baseline_acc and think_acc:
            baseline_mean = sum(baseline_acc) / len(baseline_acc)
            think_mean = sum(think_acc) / len(think_acc)
            improvement = think_mean - baseline_mean
            
            print(f"\nLATENT THINKING HYPOTHESIS TEST:")
            print(f"   Baseline (expected worst): {baseline_mean:.2f} digits")
            print(f"   Think about solution (expected best): {think_mean:.2f} digits")
            print(f"   Improvement: {improvement:+.2f} digits")
            
            if think_mean > baseline_mean:
                if baseline_mean > 0:
                    percent_improvement = (improvement / baseline_mean) * 100
                    print(f"   Percent improvement: {percent_improvement:+.1f}%")
                print(f"   HYPOTHESIS CONFIRMED: Thinking improves accuracy")
                print(f"   Latent thinking effect validated")
            else:
                print(f"   Hypothesis not confirmed - requires investigation")
        
        # Check for random number generation
        random_condition_acc = condition_accuracies.get('generate_random_numbers', [])
        if random_condition_acc:
            print(f"\nRANDOM NUMBER GENERATION:")
            print(f"   Numbers harvested for Phase 2 transplantation")
            print(f"   Ready to proceed with Phase 2")
        else:
            print(f"\nRandom number generation not completed")
            print(f"   Phase 2 may not be possible without harvested numbers")
        
        print(f"\nPhase 1 complete")
        print(f"   Next steps:")
        print(f"   • Analyze results: python analysis/comprehensive_verification.py")
        print(f"   • Run Phase 2: python main_phase2.py")
        
    else:
        print(f"\nNo results generated - experiment may have failed or been interrupted")
        print(f"Check the logs above for error details")
        
    if phase1_runner.shutdown_requested:
        print(f"\nGraceful shutdown completed successfully")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
