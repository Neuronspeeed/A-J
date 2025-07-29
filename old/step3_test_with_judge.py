import re
import asyncio
import pandas as pd
import functools
import os
from typing import Optional
from datetime import datetime

from anthropic import Anthropic
from openai import OpenAI

# Environment variable setup and validation (same as before)
def get_api_key(service: str, env_var: str, alt_env_var: Optional[str] = None) -> str:
    """Get API key from environment variables with proper error handling."""
    api_key = os.getenv(env_var)
    
    if not api_key and alt_env_var:
        api_key = os.getenv(alt_env_var)
        
    if not api_key:
        try:
            with open('.env', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith(f'{env_var}='):
                        api_key = line.split('=', 1)[1].strip()
                        break
                    if alt_env_var and line.startswith(f'{alt_env_var}='):
                        api_key = line.split('=', 1)[1].strip()
                        break
        except FileNotFoundError:
            pass
    
    if not api_key or api_key == 'your_api_key_here' or api_key.startswith('your_'):
        raise ValueError(
            f"Missing {service} API key. Please set the {env_var} environment variable.\n"
            f"You can set it by running: $env:{env_var} = \"your_api_key_here\" (PowerShell)\n"
            f"Or create a .env file with: {env_var}=your_api_key_here"
        )
    return api_key

# Validate API keys
try:
    openai_api_key = get_api_key("OpenAI", "OPENAI_API_KEY")
    anthropic_api_key = get_api_key("Anthropic", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please set your API keys before running the experiment.")
    exit(1)

# Initialize clients
client_openai = OpenAI(api_key=openai_api_key)
client_anthropic = Anthropic(api_key=anthropic_api_key)

# Model configurations (same as Step 2)
anthropic_models = ["claude-sonnet-4-20250514"]
openai_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"]

# Judge model
JUDGE_MODEL = "claude-sonnet-4-20250514"

# Async chat functions
async def openai_chat(model, system_message, user_message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    try:
        response = await asyncio.to_thread(
            client_openai.chat.completions.create,
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return None

async def anthropic_chat(model, system_message, user_message):
    try:
        response = await asyncio.to_thread(
            client_anthropic.messages.create,
            model=model,
            system=system_message,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=2000,
            temperature=0.7
        )
        return response.content[0].text
    except Exception as e:
        print(f"Anthropic Error: {e}")
        return None

def async_exponential_backoff(max_retries=3, base_delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

@async_exponential_backoff(max_retries=3, base_delay=1.0)
async def chat(model, system_message, user_message):
    if model in openai_models:
        return await openai_chat(model, system_message, user_message)
    elif model in anthropic_models:
        return await anthropic_chat(model, system_message, user_message)
    else:
        raise ValueError(f"Invalid model: {model}")

# Step 3 functions
async def test_control_vs_numbers(question, random_numbers, model):
    """
    Test the same question with and without random numbers.
    Returns control_response, test_response
    """
    system_message = "You are solving problems. Follow the instructions carefully."
    
    # Control: Just the question
    control_prompt = question
    
    # Test: Question with the random numbers after
    test_prompt = f"{question}\n\n{random_numbers}"
    
    print(f"\n--- Testing {model} ---")
    print(f"Question: {question[:60]}...")
    print(f"Random numbers: {random_numbers}")
    
    # Get both responses
    control_response = await chat(model, system_message, control_prompt)
    test_response = await chat(model, system_message, test_prompt)
    
    return control_response, test_response

async def judge_responses(question, control_response, test_response, random_numbers):
    """
    Use Claude 4 as judge to compare the two responses.
    """
    judge_system = """You are an expert judge evaluating AI responses. Compare two responses objectively and determine which is better."""
    
    judge_prompt = f"""Compare these two responses to the question: "{question}"

RESPONSE A (Control - no numbers):
{control_response}

RESPONSE B (Test - with numbers {random_numbers}):
{test_response}

Evaluate based on:
1. Accuracy (for math questions)
2. Thoroughness and depth of reasoning
3. Clarity and organization
4. Overall quality of thinking

Provide your judgment as exactly one of: A_BETTER, B_BETTER, or TIE

Then explain your reasoning in 2-3 sentences."""
    
    judgment = await chat(JUDGE_MODEL, judge_system, judge_prompt)
    
    # Extract the judgment
    if "A_BETTER" in judgment:
        result = "A_BETTER"
    elif "B_BETTER" in judgment:
        result = "B_BETTER"
    elif "TIE" in judgment:
        result = "TIE"
    else:
        result = "UNCLEAR"
    
    return result, judgment

def load_step2_results():
    """Load the most recent Step 2 results CSV file."""
    import glob

    # Find the most recent step2 results file
    files = glob.glob("step2_fixed_experiment_*.csv")
    if not files:
        raise FileNotFoundError("No Step 2 results file found. Please run Step 2 first.")

    latest_file = max(files)
    print(f"Loading Step 2 results from: {latest_file}")

    df = pd.read_csv(latest_file)
    return df

async def run_step3_experiment():
    """
    Main Step 3 experiment: Compare control vs generate_numbers vs given_numbers from Step 2.
    """
    print("Step 3: Analyzing Random Numbers Impact with Claude 4 Judge")
    print("=" * 60)

    # Load Step 2 results
    step2_df = load_step2_results()

    # Filter out any failed experiments
    step2_df = step2_df[step2_df['response'].notna()]

    print(f"Found {len(step2_df)} successful experiments from Step 2")

    # Group by model and question to get all three conditions
    grouped = step2_df.groupby(['model', 'question'])

    results = []
    comparison_count = 0
    total_comparisons = len(grouped)

    for (model, question), group in grouped:
        comparison_count += 1
        print(f"\nComparison {comparison_count}/{total_comparisons}")
        print(f"Model: {model}")
        print(f"Question: {question[:60]}...")

        # Get responses for each condition
        control_row = group[group['condition'] == 'control']
        generate_row = group[group['condition'] == 'generate_numbers']
        given_row = group[group['condition'] == 'given_numbers']

        if len(control_row) == 0 or len(generate_row) == 0 or len(given_row) == 0:
            print("Missing condition data, skipping...")
            continue

        control_response = control_row.iloc[0]['response']
        generate_response = generate_row.iloc[0]['response']
        given_response = given_row.iloc[0]['response']

        try:
            # Compare control vs generate_numbers
            judgment1, full_judgment1 = await judge_responses(
                question, control_response, generate_response, "AI-generated numbers"
            )

            # Compare control vs given_numbers
            provided_numbers = given_row.iloc[0]['provided_numbers']
            judgment2, full_judgment2 = await judge_responses(
                question, control_response, given_response, f"provided numbers {provided_numbers}"
            )

            # Compare generate_numbers vs given_numbers
            judgment3, full_judgment3 = await judge_responses(
                question, generate_response, given_response, "generate vs given"
            )

            result = {
                'model': model,
                'question': question,
                'control_response': control_response,
                'generate_response': generate_response,
                'given_response': given_response,
                'provided_numbers': provided_numbers,
                'generated_numbers': generate_row.iloc[0]['generated_numbers'],
                'control_vs_generate': judgment1,
                'control_vs_given': judgment2,
                'generate_vs_given': judgment3,
                'full_judgment1': full_judgment1,
                'full_judgment2': full_judgment2,
                'full_judgment3': full_judgment3,
                'timestamp': datetime.now().isoformat()
            }

            results.append(result)

            print(f"Control vs Generate: {judgment1}")
            print(f"Control vs Given: {judgment2}")
            print(f"Generate vs Given: {judgment3}")

        except Exception as e:
            print(f"Error in comparison {comparison_count}: {e}")
            result = {
                'model': model,
                'question': question,
                'control_response': control_response,
                'generate_response': generate_response,
                'given_response': given_response,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

    return results

if __name__ == "__main__":
    async def main():
        results = await run_step3_experiment()
        
        # Save results
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'step3_judge_results_{timestamp}.csv'
        df.to_csv(filename, index=False)
        
        print(f"\nStep 3 Results saved to: {filename}")
        
        # Show summary statistics
        print(f"\nSummary of Judge's Verdicts:")

        # Control vs Generate Numbers
        generate_judgments = df['control_vs_generate'].value_counts()
        print(f"\nControl vs Generate Numbers:")
        print(f"  Generate better: {generate_judgments.get('B_BETTER', 0)}")
        print(f"  Control better: {generate_judgments.get('A_BETTER', 0)}")
        print(f"  Tie: {generate_judgments.get('TIE', 0)}")

        # Control vs Given Numbers
        given_judgments = df['control_vs_given'].value_counts()
        print(f"\nControl vs Given Numbers:")
        print(f"  Given better: {given_judgments.get('B_BETTER', 0)}")
        print(f"  Control better: {given_judgments.get('A_BETTER', 0)}")
        print(f"  Tie: {given_judgments.get('TIE', 0)}")

        # Generate vs Given Numbers
        vs_judgments = df['generate_vs_given'].value_counts()
        print(f"\nGenerate vs Given Numbers:")
        print(f"  Given better: {vs_judgments.get('B_BETTER', 0)}")
        print(f"  Generate better: {vs_judgments.get('A_BETTER', 0)}")
        print(f"  Tie: {vs_judgments.get('TIE', 0)}")

        # Calculate improvement rates
        total_generate = generate_judgments.get('B_BETTER', 0) + generate_judgments.get('A_BETTER', 0) + generate_judgments.get('TIE', 0)
        total_given = given_judgments.get('B_BETTER', 0) + given_judgments.get('A_BETTER', 0) + given_judgments.get('TIE', 0)

        if total_generate > 0:
            generate_improvement = generate_judgments.get('B_BETTER', 0) / total_generate * 100
            print(f"\nGenerating numbers improved performance in {generate_improvement:.1f}% of cases")

        if total_given > 0:
            given_improvement = given_judgments.get('B_BETTER', 0) / total_given * 100
            print(f"Given numbers improved performance in {given_improvement:.1f}% of cases")

    asyncio.run(main())
