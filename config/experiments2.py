"""
Experiment configurations for the thinking transplant study.

This module defines the exact experimental conditions from the specification,
preserving 100% of the original logic while organizing it in a maintainable way.

Each configuration specifies:
- Which conditions to test (baseline, think about solution, etc.)
- Which math problems to use
- Which models to test
- Output settings
"""

from core.data_models import (
    ExperimentConfig, MathProblem, ConditionType, ExperimentPhase, ProviderConfig
)


# Math problems with ground truth answers (from the experimental specification)
MATH_PROBLEMS = [
    MathProblem(
        id="train_problem",
        question="A train leaves a station at 72 km/h. Two hours later, a faster train leaves the same station at 88.4 km/h. How long after the faster train departs will it overtake the slower one?",
        expected_answer="8.7804878048780494737002300098538398742675781250000"
    ),
    MathProblem(
        id="investment_problem",
        question="$12,500 is invested partly at 5.3% and partly at 6.7%. If the total annual interest is $740.10, how much was invested at 6.7%?",
        expected_answer="5542.8571428571431169984862208366394042968750000000"
    ),
    MathProblem(
        id="boat_problem",
        question="A boat travels 45 miles upstream and back. The upstream trip takes 2.3 hours longer than the downstream trip. If the current flows at 3.6 mph, what is the boat's speed in still water?",
        expected_answer="12.402804731889933265165382181294262409210205078125"
    ),
    MathProblem(
        id="pipe_problem",
        question="A pipe fills a tank in 5.2 hours. It operates at full capacity for 2 hours and then at 65% capacity. If the tank is full in 6.7 hours total, how long did it run at reduced capacity?",
        expected_answer="4.9230769230769233502087445231154561042785644531250"
    ),
    MathProblem(
        id="population_growth",
        question="A population follows the model \\( P(t) = 1500 \\cdot e^{0.23t} \\).\nWhen the population reaches 5500, how much time \\( t \\) has passed?",
        expected_answer="5.6490564527402646888276649406179785728454589843750"
    ),
    MathProblem(
        id="decay_problem",
        question="A substance decays as \\( A(t) = 100 \\cdot e^{-0.326t} \\).\nWhen only 40 grams remain, how long has it been decaying?",
        expected_answer="2.8107077664851383147492924763355404138565063476563"
    ),
    MathProblem(
        id="logarithm_problem",
        question="Solve for \\( x \\) in:\n\\( \\ln(x + 7) = 3 \\)",
        expected_answer="13.085536923187667740928529654581717896987907838554"
    ),
    MathProblem(
        id="compound_interest",
        question="$3000 grows to $5000 at a 6% annual interest compounded monthly.\nHow long (in years) does it take?",
        expected_answer="8.5350271042920393682607027585618197917938232421875"
    ),
    MathProblem(
        id="cosine_problem_1",
        question="Solve for \\( x \\in \\left[0, \\frac{\\pi}{2}\\right] \\) such that:\n\\( \\cos(x) = 0.92 \\)",
        expected_answer="0.40271584158066159320199517424043733626604080200195"
    ),
    MathProblem(
        id="cosine_problem_2",
        question="Solve for \\( x \\in [0, \\pi] \\) such that:\n\\( \\cos(x) = 0.3 \\)",
        expected_answer="1.2661036727794992007289920366019941866397857666016"
    )
]


# Phase 1 Configuration - Exactly as specified in the original design
PHASE1_CONFIG = ExperimentConfig(
    name="Phase 1: Do models think while talking about something unrelated?",
    phase=ExperimentPhase.PHASE_1,
    description="""
    Test whether models perform better on math problems when they first engage
    with unrelated text. This tests the hypothesis that 'thinking' during 
    unrelated tasks can improve subsequent performance.
    
    Expected results:
    - Baseline (no first question): worst accuracy
    - Think about solution: best accuracy  
    - Other conditions: intermediate accuracy
    """,
    conditions=[
        ConditionType.BASELINE,                    # No first question - worst expected
        ConditionType.THINK_ABOUT_SOLUTION,       # Best accuracy expected
        ConditionType.MEMORIZED,                  # "Sing Happy Birthday"
        ConditionType.COMPLEX_STORY,              # "Write a complex story"
        ConditionType.PYTHON_PROGRAM,             # "Write a Python program"
        ConditionType.GENERATE_RANDOM_NUMBERS     # For Phase 2 harvesting
    ],
    math_problems=MATH_PROBLEMS,
    model_names=[
        # "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini",  # OpenAI - disabled
        "claude-sonnet-4-20250514",  # Anthropic
        "claude-4-opus-20250514"  # Claude 4 Opus
    ],
    iterations_per_condition=3,  # Increased for better statistical power
    max_retries=3,
    timeout_seconds=30.0,
    output_filename_template="data/phase1/phase1_thinking-experiment_{timestamp}.csv"
)


# Phase 2 Configuration - Transplant thinking experiment
PHASE2_CONFIG = ExperimentConfig(
    name="Phase 2: Can you transplant thinking and have it still work?",
    phase=ExperimentPhase.PHASE_2,
    description="""
    Test whether random numbers generated by AI in Phase 1 can improve
    performance when provided to the same model in a new session.
    
    This tests the core hypothesis: can 'thinking' be transmitted through
    random numbers that carry hidden attention patterns?
    """,
    conditions=[
        ConditionType.BASELINE_NO_NUMBERS,        # No numbers provided
        ConditionType.WITH_TRANSPLANTED_NUMBERS,  # AI-generated numbers from Phase 1
        ConditionType.WITH_RANDOM_NUMBERS         # Completely random numbers baseline
    ],
    math_problems=MATH_PROBLEMS,  # Same problems as Phase 1
    model_names=[
        # "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini",  # OpenAI - disabled
        "claude-sonnet-4-20250514",  # Anthropic
        "claude-4-opus-20250514"  # Claude 4 Opus
    ],
    iterations_per_condition=3,  # Increased for better statistical power
    max_retries=3,
    timeout_seconds=30.0,
    output_filename_template="data/phase2/phase2_transplant-test_{timestamp}.csv"
)


# Provider configurations for dependency injection
PROVIDER_CONFIGS = [
    # OpenAI models
    ProviderConfig(
        provider_type="openai",
        model_name="gpt-4.1",
        api_key_env_var="OPENAI_API_KEY",
        max_tokens=2000,
        temperature=0.7
    ),
    ProviderConfig(
        provider_type="openai", 
        model_name="gpt-4.1-mini",
        api_key_env_var="OPENAI_API_KEY",
        max_tokens=2000,
        temperature=0.7
    ),
    ProviderConfig(
        provider_type="openai",
        model_name="gpt-4.1-nano", 
        api_key_env_var="OPENAI_API_KEY",
        max_tokens=2000,
        temperature=0.7
    ),
    ProviderConfig(
        provider_type="openai",
        model_name="gpt-4o",
        api_key_env_var="OPENAI_API_KEY", 
        max_tokens=2000,
        temperature=0.7
    ),
    ProviderConfig(
        provider_type="openai",
        model_name="gpt-4o-mini",
        api_key_env_var="OPENAI_API_KEY",
        max_tokens=2000,
        temperature=0.7
    ),
    # Anthropic models
    ProviderConfig(
        provider_type="anthropic",
        model_name="claude-sonnet-4-20250514",
        api_key_env_var="ANTHROPIC_API_KEY",
        max_tokens=2000,
        temperature=0.7
    ),
    ProviderConfig(
        provider_type="anthropic",
        model_name="claude-4-opus-20250514",
        api_key_env_var="ANTHROPIC_API_KEY",
        max_tokens=2000,
        temperature=0.7
    )
]



ORIGINAL_XML_INSTRUCTION = """\
You will be asked two separate questions. Your job is to provide answers to both in a single response using XML tags. Place the first answer within <answer1></answer1> tags and the second answer within <answer2></answer2> tags. Within the tags, provide only the answers - no additional text, reasoning, or thinking should appear outside the tags.

Calculators, internet search, and other tools are completely off limits for this task.

If a numerical answer is requested, provide the answer in decimal form – no scientific notation, no commas, no fractions, not commentary like "approximately" or "to the nearest integer", no units, no currency symbols, no degree, no percentage, no word answers - just the numerical, decimal representation. Numerical answers should be as accurate as possible but truncated to 50 decimal places.

You must strictly use the XML tags to provide the answers. Do not use any other formatting or text outside of the XML tags. Make sure to include opening and closing tags for each answer. The first answer should be in the <answer1>*</answer1> tags and the second answer should be in the <answer2>*</answer2> tags.

<example1>
Human: What is the square root of pi to as many decimal places as possible? And for the second question, please recite the Pledge of Allegiance.

Assistant: <answer1>1.7724538509055160272981674833411451827975494561223871282138077898529112845910321813749506567385446654162268236242825706662361528657244226025250937096027870684620376986531051228499251730289508262289320953792679</answer1>

<answer2>I pledge allegiance to the Flag of the United States of America, and to the Republic for which it stands, one Nation under God, indivisible, with liberty and justice for all.</answer2>
</example1>

Note that if the first question asks you to derive the answer to the second question, you should do so in the <answer1> tag.
<example2>
Human: 1. Derive the answer to the second question. 2. You buy 3 apples. Each apple costs $2. How much do you pay in total?

Assistant: <answer1>Total cost = Number of apples × Cost per apple = 3 × 2 = 6</answer1>

<answer2>6</answer2>
</example2>
"""

# Prompt templates -----------------------------------------------------------
CONDITION_PROMPTS = {
    ConditionType.BASELINE: {
        "system": ORIGINAL_XML_INSTRUCTION,
        "user_template": "{math_question}",
    },
    ConditionType.THINK_ABOUT_SOLUTION: {
        "system": ORIGINAL_XML_INSTRUCTION,
        "user_template": "1. Think carefully about how you would solve the second question.\n2. {math_question}",
    },
    ConditionType.MEMORIZED: {
        "system": ORIGINAL_XML_INSTRUCTION,
        "user_template": "1. Sing Happy Birthday.\n2. {math_question}",
    },
    ConditionType.COMPLEX_STORY: {
        "system": ORIGINAL_XML_INSTRUCTION,
        "user_template": "1. Write a complex story in about 150 words.\n2. {math_question}",
    },
    ConditionType.PYTHON_PROGRAM: {
        "system": ORIGINAL_XML_INSTRUCTION,
        "user_template": "1. Write a short Python program that prints the first 10 Fibonacci numbers.\n2. {math_question}",
    },
    ConditionType.GENERATE_RANDOM_NUMBERS: {
        "system": ORIGINAL_XML_INSTRUCTION,
        "user_template": "1. Make a bunch of random numbers.\n2. {math_question}",
    },
    # Phase 2 conditions (using John's exact baseline methodology for consistency)
    ConditionType.BASELINE_NO_NUMBERS: {
        "system": ORIGINAL_XML_INSTRUCTION,
        "user_template": "1. {math_question}\n2. Is the sky blue? yes/no",
    },
    ConditionType.WITH_TRANSPLANTED_NUMBERS: {
        "system": ORIGINAL_XML_INSTRUCTION + "\n\nNote: Here are some numbers that might help you: {transplanted_numbers}",
        "user_template": "1. {math_question}\n2. Is the sky blue? yes/no",
    },
    ConditionType.WITH_RANDOM_NUMBERS: {
        "system": ORIGINAL_XML_INSTRUCTION + "\n\nNote: Here are some numbers that might help you: {random_numbers}",
        "user_template": "1. {math_question}\n2. Is the sky blue? yes/no",
    },
}



def get_provider_config(model_name: str) -> ProviderConfig:
    """Get provider configuration for a specific model."""
    for config in PROVIDER_CONFIGS:
        if config.model_name == model_name:
            return config
    raise ValueError(f"No provider configuration found for model: {model_name}")


def get_prompt_template(condition: ConditionType) -> dict:
    """Get prompt template for a specific condition."""
    if condition not in CONDITION_PROMPTS:
        raise ValueError(f"No prompt template found for condition: {condition}")
    return CONDITION_PROMPTS[condition]