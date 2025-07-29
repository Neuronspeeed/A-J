"""
Main entry point for Phase 1 of the thinking transplant experiment.

This script implements the Composition Root pattern, where all dependencies
are constructed and injected. This is the ONLY place where concrete classes
are instantiated, following the Dependency Inversion Principle.

Phase 1 tests: "Do the models think while talking about something unrelated?"

Usage:
    uv run python main_phase1.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from core.llm_providers import create_provider, LLMProviderError
from core.persistence import CsvResultWriter
from engine.experiment_runner import ExperimentRunner
from config.experiments import PHASE1_CONFIG, get_provider_config


async def main():
    """
    Main function implementing the Composition Root pattern.

    This function:
    1. Validates environment setup
    2. Creates concrete dependencies
    3. Injects them into the experiment engine
    4. Runs the experiment
    5. Reports results
    """
    # Check for test mode
    test_mode = "--test-mode" in sys.argv

    print("Phase 1: Do the models think while talking about something unrelated?")
    print("=" * 70)

    if test_mode:
        print("🧪 RUNNING IN TEST MODE (reduced scope)")
    else:
        print("🚀 RUNNING FULL EXPERIMENT")
    
    # Validate that we have API keys
    try:
        # Test provider creation to validate API keys early
        test_config = get_provider_config("gpt-4o")
        test_provider = create_provider(test_config)
        print("✅ API keys validated")
    except Exception as e:
        print(f"❌ API key validation failed: {e}")
        print("\nPlease ensure you have set the following environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY (or CLAUDE_API_KEY)")
        return 1
    
    # Create experiment configuration (copy so we can modify for test mode)
    config = PHASE1_CONFIG.model_copy()

    if test_mode:
        # Reduce scope for testing
        config.math_problems = config.math_problems[:2]  # Just 2 problems
        config.iterations_per_condition = 1  # Just 1 iteration
        config.model_names = ["claude-sonnet-4-20250514"]  # Just one model
        print("🧪 Test mode: 2 problems, 1 iteration, 1 model")

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = config.output_filename_template.format(timestamp=timestamp)

    print(f"📁 Results will be saved to: {output_filename}")
    print(f"🧪 Testing {len(config.model_names)} models")
    print(f"📋 Testing {len(config.conditions)} conditions")
    print(f"🔢 Testing {len(config.math_problems)} math problems")

    total_trials = (
        len(config.model_names) *
        len(config.conditions) *
        len(config.math_problems) *
        config.iterations_per_condition
    )
    print(f"📊 Total trials: {total_trials}")
    
    # Create result writer (dependency injection)
    writer = CsvResultWriter(output_filename)
    
    all_results = []
    
    try:
        # Run experiment for each model
        for model_name in config.model_names:
            print(f"\n🤖 Testing model: {model_name}")
            
            try:
                # Create provider for this model (dependency injection)
                provider_config = get_provider_config(model_name)
                provider = create_provider(provider_config)
                
                # Create experiment runner (dependency injection)
                runner = ExperimentRunner(provider=provider, writer=writer)
                
                # Create config for this specific model
                model_config = PHASE1_CONFIG.model_copy(deep=True)
                model_config.model_names = [model_name]  # Test only this model
                
                # Run experiment
                results = await runner.run_experiment(model_config)
                all_results.append(results)
                
                print(f"✅ Completed {model_name}: {results.successful_trials}/{results.total_trials} successful")
                
            except LLMProviderError as e:
                print(f"❌ Provider error for {model_name}: {e}")
                continue
            except Exception as e:
                print(f"❌ Unexpected error for {model_name}: {e}")
                continue
    
    finally:
        # Finalize output file
        final_filename = writer.finalize()
        print(f"\n📁 Results saved to: {final_filename}")
    
    # Print summary
    if all_results:
        print(f"\n{'='*70}")
        print("PHASE 1 SUMMARY")
        print(f"{'='*70}")
        
        total_successful = sum(r.successful_trials for r in all_results)
        total_trials = sum(r.total_trials for r in all_results)
        
        print(f"Overall success rate: {total_successful}/{total_trials} ({100*total_successful/total_trials:.1f}%)")
        
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
        
        # Test hypothesis
        baseline_acc = condition_accuracies.get('baseline', [0])
        think_acc = condition_accuracies.get('think_about_solution', [0])
        
        if baseline_acc and think_acc:
            baseline_mean = sum(baseline_acc) / len(baseline_acc)
            think_mean = sum(think_acc) / len(think_acc)
            
            print(f"\nHypothesis Test:")
            print(f"  Baseline (expected worst): {baseline_mean:.2f}")
            print(f"  Think about solution (expected best): {think_mean:.2f}")
            
            if think_mean > baseline_mean:
                print(f"  ✅ HYPOTHESIS CONFIRMED: Thinking improves accuracy!")
            else:
                print(f"  ❌ Hypothesis not confirmed")
        
        print(f"\n🎯 Phase 1 complete! Ready to run Phase 2 with:")
        print(f"   uv run python main_phase2.py")
        
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
