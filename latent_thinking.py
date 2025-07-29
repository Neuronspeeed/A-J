import re
import asyncio
import pandas as pd
import functools
import os
from typing import Optional

from anthropic import Anthropic
from openai import OpenAI

# Environment variable setup and validation
def get_api_key(service: str, env_var: str, alt_env_var: Optional[str] = None) -> str:
    """Get API key from environment variables with proper error handling."""
    api_key = os.getenv(env_var)
    
    # Try alternative environment variable if provided
    if not api_key and alt_env_var:
        api_key = os.getenv(alt_env_var)
        
    # Try to read from .env file if available
    if not api_key:
        try:
            with open('.env', 'r') as f:
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

# Validate API keys are available
try:
    openai_api_key = get_api_key("OpenAI", "OPENAI_API_KEY")
    anthropic_api_key = get_api_key("Anthropic", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please set your API keys before running the experiment.")
    exit(1)

# Initialize the OpenAI client
client_openai = OpenAI(api_key=openai_api_key)

# Initialize the Anthropic client
client_anthropic = Anthropic(api_key=anthropic_api_key)

PRINT_USER_MESSAGE = False
PRINT_RESPONSE = False

async def openai_chat(model, system_message, user_message):
    """
    Call OpenAI chat completion and return just the assistant's text.
    """
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
    """
    Call Anthropic chat completion and return just the assistant's text.
    """
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

def extract_answers(xml_response):
    """
    Extracts answers from a string containing <answer1>...</answer1> and <answer2>...</answer2> tags.
    Returns a list of answer strings in order. If a tag is not closed, extracts up to the end and sets the other to None.
    """
    pattern = re.compile(r"<answer(\d+)>((?:.|\n)*?)</answer\1>", re.IGNORECASE)
    matches = pattern.findall(xml_response)
    answers = [None, None]  # For <answer1>, <answer2>
    for num, ans in matches:
        idx = int(num) - 1
        if idx in (0, 1):
            answers[idx] = ans.strip()
    # If only one answer found, check for an unclosed tag for the other
    if answers[0] is None:
        m = re.search(r"<answer1>(.*)", xml_response, re.IGNORECASE | re.DOTALL)
        if m:
            # If <answer1> exists but not closed, take everything after <answer1>
            after = m.group(1)
            # If <answer2> starts after <answer1>, only take up to <answer2>
            split = re.split(r"<answer2>", after, flags=re.IGNORECASE)
            answers[0] = split[0].strip() if split else after.strip()
    if answers[1] is None:
        m = re.search(r"<answer2>(.*)", xml_response, re.IGNORECASE | re.DOTALL)
        if m:
            after = m.group(1)
            # If <answer1> starts after <answer2>, only take up to <answer1>
            split = re.split(r"<answer1>", after, flags=re.IGNORECASE)
            answers[1] = split[0].strip() if split else after.strip()
    return answers

# Anthropic – Claude 4 Family (released May 22, 2025)
anthropic_models = [
    # "claude-opus-4", # not accessible
    "claude-sonnet-4-20250514",
]

# OpenAI – Latest GPT Models
openai_models = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]

def async_exponential_backoff(max_retries=5, base_delay=1.0):
    """
    Decorator for async functions to retry with exponential backoff on exception.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

@async_exponential_backoff(max_retries=10, base_delay=1.0)
async def chat(model, system_message, user_message):
    if model in openai_models:
        return await openai_chat(model, system_message, user_message)
    elif model in anthropic_models:
        return await anthropic_chat(model, system_message, user_message)
    else:
        raise ValueError(f"Invalid model: {model}")


def compare_decimal_strings(expected, actual):
    """
    Compares two strings representing decimal numbers and counts the number of correct digits (ignoring the decimal point)
    from the most significant digit until the first mismatch.
    
    Args:
        expected (str): The expected number as a string (e.g., '5.264')
        actual (str): The actual number as a string (e.g., '5.164')
    Returns:
        int: The number of correct digits from the start (ignoring the decimal point)
    
    Examples:
        compare_decimal_strings('5.264', '4.264') -> 0
        compare_decimal_strings('5.264', '5.164') -> 1
        compare_decimal_strings('5.264', '5.263') -> 3
        compare_decimal_strings('12.345', '12.349') -> 4
        compare_decimal_strings('12.345', '12.345') -> 5
    """
    # Remove decimal points
    exp_digits = expected.replace('.', '')
    act_digits = actual.replace('.', '')
    
    # Compare digit by digit
    correct = 0
    for e, a in zip(exp_digits, act_digits):
        if e == a:
            correct += 1
        else:
            break
    return correct

async def compare_model_on_questions(math_question, math_expected_answer, model, filler_question=None):
    """
    Run the given model on the two-question prompt and return a dict with 'answer' and 'digits_correct'.
    model: model_name (string)
    """
    system_message = """\
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
    if filler_question is None:
        filler_question = "Is the sky blue? yes/no"
        math_first = True
    else:
        math_first = False
    if math_first:
        user_message = f"1. {math_question}\n2. {filler_question}"
    else:
        user_message = f"1. {filler_question}\n2. {math_question}"
    important_answer = 0 if math_first else 1

    if PRINT_USER_MESSAGE:
        print(f"User message: {user_message}")
        print()

    response = await chat(model, system_message, user_message)
    if PRINT_RESPONSE:
        print(f"Model: {model}")
        print(f"Response: {response}")
        print()
    answers = extract_answers(response or "")
    if len(answers) > important_answer:
        model_answer = answers[important_answer]
        digits_correct = compare_decimal_strings(math_expected_answer, model_answer)
    else:
        print(f"Failed response from {model}: {response}")
        raise Exception(f"Failed to get answer from {model}")
    return {"answer": model_answer, "digits_correct": digits_correct}

NUM_CONCURRENT_TASKS = 20

async def evaluate_models_on_math_questions(filler_questions, math_questions_with_answers, models, num_iterations=1):
    """
    Given lists of filler_questions, math_questions_with_answers [(question, answer)], and models,
    run all combinations asynchronously and return a list of dicts:
    { 'model': ..., 'question': ..., 'answer': ..., 'num_digits_correct': ..., 'filler_question': ..., 'expected_answer': ..., "error": ..., 'iteration': ..., 'num_iterations': ... }
    """
    semaphore = asyncio.Semaphore(NUM_CONCURRENT_TASKS)
    total_to_do = len(filler_questions) * len(math_questions_with_answers) * len(models) * num_iterations
    progress = {'done': 0}
    async def sem_task(math_question, math_answer, model_name, filler_question, iteration):
        async with semaphore:
            error = None
            try:
                r = await compare_model_on_questions(math_question, math_answer, model_name, filler_question)
            except Exception as e:
                r = None
                error = str(e)
            result = {
                'model': model_name,
                'question': math_question,
                'filler_question': filler_question,
                'expected_answer': math_answer,
                'returned_answer': r['answer'] if r else None,
                'num_digits_correct': r['digits_correct'] if r else None,
                'error': error,
                'iteration': iteration + 1,  # 1-based
                'num_iterations': num_iterations
            }
            progress['done'] += 1
            print(f"{progress['done']}/{total_to_do}")
            return result
    tasks = []
    for filler_question in filler_questions:
        for math_question, math_answer in math_questions_with_answers:
            for model_name in models:
                for iteration in range(num_iterations):
                    tasks.append(sem_task(math_question, math_answer, model_name, filler_question, iteration))
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: (x['model'], x['question'], str(x['filler_question']), x['iteration']))

    return results

if __name__ == "__main__":
    async def main():
        math_questions_with_answers = [
            (
                "A train leaves a station at 72 km/h. Two hours later, a faster train leaves the same station at 88.4 km/h. How long after the faster train departs will it overtake the slower one?", 
                "8.7804878048780494737002300098538398742675781250000"
            ),
            (
                "$12,500 is invested partly at 5.3% and partly at 6.7%. If the total annual interest is $740.10, how much was invested at 6.7%?", 
                "5542.8571428571431169984862208366394042968750000000"
            ),
            (
                "A boat travels 45 miles upstream and back. The upstream trip takes 2.3 hours longer than the downstream trip. If the current flows at 3.6 mph, what is the boat's speed in still water?",
                "12.402804731889933265165382181294262409210205078125"
            ),
            (
                "A pipe fills a tank in 5.2 hours. It operates at full capacity for 2 hours and then at 65% capacity. If the tank is full in 6.7 hours total, how long did it run at reduced capacity?",
                "4.9230769230769233502087445231154561042785644531250"
            ),
            (
                "A population follows the model \\( P(t) = 1500 \\cdot e^{0.23t} \\).\nWhen the population reaches 5500, how much time \\( t \\) has passed?",
                "5.6490564527402646888276649406179785728454589843750"
            ),
            (
                "A substance decays as \\( A(t) = 100 \\cdot e^{-0.326t} \\).\nWhen only 40 grams remain, how long has it been decaying?",
                "2.8107077664851383147492924763355404138565063476563"
            ),
            (
                "Solve for \\( x \\) in:\n\\( \\ln(x + 7) = 3 \\)",
                "13.085536923187667740928529654581717896987907838554"
            ),
            (
                "$3000 grows to $5000 at a 6% annual interest compounded monthly.\nHow long (in years) does it take?",
                "8.5350271042920393682607027585618197917938232421875"
            ),
            (
                "Solve for \\( x \\in \\left[0, \\frac{\\pi}{2}\\right] \\) such that:\n\\( \\cos(x) = 0.92 \\)",
                "0.40271584158066159320199517424043733626604080200195"
            ),
            (
                "Solve for \\( x \\in [0, \\pi] \\) such that:\n\\( \\cos(x) = 0.3 \\)",
                "1.2661036727794992007289920366019941866397857666016"
            ),
        ]

        filler_questions = [
            None,
            "What is the first paragraph of the declaration of independence?",
            "Why was the Declaration of Independence written? (~100 words)",
            "What is the derivation of the second question?",
        ]
        
        
        models = openai_models + anthropic_models

        num_iterations = 5

        # Use the new function
        results = await evaluate_models_on_math_questions(filler_questions, math_questions_with_answers, models, num_iterations=num_iterations)
        # Convert results to pandas DataFrame
        df = pd.DataFrame(results)
        
        # Write DataFrame to CSV file
        print(f"Writing results to results.csv")
        df.to_csv('results.csv', index=False)

    asyncio.run(main())