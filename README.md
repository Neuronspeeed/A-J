# Latent Thinking Experiment

This project runs a multi-phase experiment to test if AI models can "think" about a math problem while distracted by a second task. It measures how different kinds of tasks affect the model's accuracy on math problems.

## Core Questions

1.  **Phase 1**: Does a model's performance on a math problem change if it first processes an unrelated piece of text (a "filler question")?
2.  **Phase 2**: Can the "thinking" process, as captured by random numbers generated in Phase 1, be transplanted into a new session to improve performance on the same problem?

## How the Code is Organized

The project is organized into separate parts to make the code clean and easy to work with.

-   **`core/`**: Contains the project's main components:
    -   `data_models.py`: Defines the data structures for the project (e.g., `TrialResult`, `ExperimentConfig`).
    -   `llm_providers.py`: Contains the code for interacting with different LLM providers like OpenAI and Anthropic.
    -   `persistence.py`: Contains the code for saving experiment results to files.
-   **`engine/`**: Contains `ExperimentRunner`, the main class that runs the experiment. It can work with any LLM provider or storage method defined in `core`.
-   **`config/`**: Defines the exact experimental conditions (prompts, problems, models) for each phase. This separates the experiment's configuration from its logic.
-   **`main_*.py`**: Scripts to start the experiments. They set up the necessary components and pass them to the main experiment runner.
-   **`tests/`**: Contains automated tests to verify the experiment logic works correctly without making real API calls.

## How to Run the Experiment

### Prerequisites

-   Python 3.13+
-   An active virtual environment (e.g., via `uv venv`)

### Installation

1.  Install dependencies:
    ```bash
    uv sync
    ```

2.  Set up API keys:
    ```bash
    # Copy the example file and edit it
    cp env.example .env
    
    # Edit .env with your actual API keys:
    # OPENAI_API_KEY=sk-your-openai-api-key-here
    # ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
    ```

### Running the Experiments

The two phases must be run in order.

**1. Run Phase 1 (Data Generation)**

This runs the filler-question experiment and generates the data needed for Phase 2.

```bash
python main_phase1.py
```

This will create a results file named `phase1_results_<timestamp>.csv`.

**2. Run Phase 2 (Transplant Thinking)**

This automatically finds the latest Phase 1 results, harvests the AI-generated numbers, and runs the transplant experiment.

```bash
python main_phase2.py
```

This will create a results file named `phase2_results_<timestamp>.csv`.

## Testing

To run the automated tests:

```bash
pytest
```

To run tests with coverage report:

```bash
pytest --cov=core --cov=engine --cov-report=term-missing
```

The test suite runs automatically on all code changes to ensure reliability and prevent regressions.

## How to Extend the Experiment

The organized design makes extensions simple and safe.

-   **To Add a New Model (e.g., Gemini)**:
    1.  Add a `GeminiProvider` class in `core/llm_providers.py`.
    2.  Add its configuration to `config/experiments.py`.
    3.  Add the new model name to the `model_names` list in the desired experiment config.

-   **To Add a New Math Problem**:
    1.  Add a new `MathProblem` to the `MATH_PROBLEMS` list in `config/experiments.py`.

-   **To Add a New Condition**:
    1.  Add a new condition type in `core/data_models.py`.
    2.  Add a matching prompt template in `config/experiments.py`.
    3.  Add the new condition to the `conditions` list in the desired experiment config.

## Project Structure

```
.
├── config/
│   ├── experiments.py         # Defines Phase 1 & 2 configurations
│   └── __init__.py
├── core/
│   ├── data_models.py         # Data structures for the experiment
│   ├── llm_providers.py       # Code for calling different LLMs
│   ├── persistence.py         # Code for saving results
│   └── __init__.py
├── engine/
│   ├── experiment_runner.py   # Main experiment logic
│   └── __init__.py
├── tests/
│   ├── conftest.py            # Test helpers
│   └── test_experiment_runner.py # Automated tests for the engine
├── main_phase1.py             # Script to run Phase 1
├── main_phase2.py             # Script to run Phase 2
├── .env.example               # Example environment file
├── pyproject.toml             # Dependencies and project configuration
└── README.md                  # This documentation
```
