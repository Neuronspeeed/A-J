# A-J Latent Thinking Experiment

Research codebase testing whether AI models can "think" about problems while working on unrelated tasks.

## Project Structure

```
core/               # Data models, LLM providers, utilities
engine/             # Experiment orchestration
config/             # Experimental configurations
analysis/           # Analysis scripts and reports
data/               # Experimental results (phase1/, phase2/)
tests/              # Automated tests
```

## Quick Start

```bash
uv sync                           # Install dependencies
cp .env.example .env              # Configure API keys
python main_phase1.py             # Run Phase 1
python main_phase2.py             # Run Phase 2
python analysis/comprehensive_verification.py  # Analyze results
```

## Experimental Design

### Phase 1: Latent Thinking Effect
Tests if engaging with unrelated tasks before math problems affects accuracy.

**Conditions**: baseline, think_about_solution, memorized, complex_story, python_program, generate_random_numbers

**Result**: +89.4% improvement with "think about solution" (p=0.0158)

### Phase 2: Number Injection Effect
Tests if AI-generated numbers can transfer "thinking" to fresh sessions.

**Conditions**: baseline_no_numbers, with_transplanted_numbers, with_random_numbers

**Result**: Random numbers (+39.9%) outperformed AI numbers (+33.7%). Effect is attention-based, not thinking transfer.

## Key Components

| File | Purpose |
|------|---------|
| `core/data_models.py` | Pydantic models: MathProblem, TrialResult, ExperimentConfig |
| `core/llm_providers.py` | OpenAI/Anthropic API interfaces with retry logic |
| `engine/experiment_runner.py` | Trial orchestration and result collection |
| `config/experiments2.py` | Production config with enhanced system prompts |

## Accuracy Metric

Counts correct digits from start of numerical answer (50-decimal precision).
- Example: Expected "8.7804", Got "8.7823" → 3 digits correct

## Models Tested

- **OpenAI**: gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini
- **Anthropic**: claude-sonnet-4-20250514, claude-4-opus-20250514

## Key Findings

1. **Latent Thinking**: Metacognitive priming improves math performance by 89%
2. **Number Injection**: Any numerical context activates mathematical reasoning circuits
3. **System Prompts**: Behavioral training examples in prompts dramatically improve baseline performance
