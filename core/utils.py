"""
Shared utility functions for the thinking transplant experiment.

This module contains pure functions that are used across multiple components,
following the DRY principle and ensuring consistency.
"""

import re
from typing import Optional, List


def compare_decimal_strings(expected: str, actual: str) -> int:
    """
    Compare decimal strings and count correct digits from the start.
    
    This is the core accuracy metric for the experiment, counting how many
    digits match from the most significant digit until the first mismatch.
    
    Args:
        expected: Ground truth answer as string
        actual: Model's answer as string
        
    Returns:
        Number of correct digits from the start
        
    Examples:
        >>> compare_decimal_strings("123.456", "123.456")
        6
        >>> compare_decimal_strings("123.456", "123.999")
        3
        >>> compare_decimal_strings("123.456", "999.456")
        0
    """
    if not expected or not actual:
        return 0
    
    # Remove decimal points for digit-by-digit comparison
    exp_digits = expected.replace('.', '')
    act_digits = actual.replace('.', '')
    
    correct = 0
    for e, a in zip(exp_digits, act_digits):
        if e == a:
            correct += 1
        else:
            break
    
    return correct


def extract_xml_answers(response: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract answers from <answer1> and <answer2> XML tags.
    
    This implements the exact parsing logic from your friend's specification.
    
    Args:
        response: Full model response containing XML tags
        
    Returns:
        Tuple of (first_answer, second_answer), either can be None
        
    Examples:
        >>> extract_xml_answers("<answer1>Hello</answer1><answer2>42</answer2>")
        ("Hello", "42")
        >>> extract_xml_answers("No tags here")
        (None, None)
    """
    if not response:
        return None, None
    
    pattern = re.compile(r"<answer(\d+)>((?:.|\n)*?)</answer\1>", re.IGNORECASE)
    matches = pattern.findall(response)
    
    answers = [None, None]
    for num, content in matches:
        idx = int(num) - 1
        if idx in (0, 1):
            answers[idx] = content.strip()
    
    return answers[0], answers[1]


def extract_random_numbers(text: str) -> Optional[List[int]]:
    """
    Extract random numbers from AI response.
    
    Looks for sequences of large numbers (6+ digits) that the AI generated
    when asked to "make a bunch of random numbers".
    
    Args:
        text: Text containing potential random numbers
        
    Returns:
        List of integers if found, None otherwise
        
    Examples:
        >>> extract_random_numbers("Numbers: 1234567890, 9876543210")
        [1234567890, 9876543210]
        >>> extract_random_numbers("No big numbers here: 1, 2, 3")
        None
    """
    if not text:
        return None
    
    # Look for sequences of large numbers (6+ digits)
    numbers = re.findall(r'\b\d{6,}\b', text)
    if len(numbers) >= 3:  # Need at least 3 numbers
        return [int(n) for n in numbers[:10]]  # Take up to 10 numbers
    
    return None


def extract_numerical_answer(response: str) -> Optional[str]:
    """
    Extract numerical answer from response text.
    
    Looks for common answer patterns and returns the numerical value
    as a string to preserve precision.
    
    Args:
        response: Model response containing a numerical answer
        
    Returns:
        Numerical answer as string, or None if not found
        
    Examples:
        >>> extract_numerical_answer("The answer is 42.5")
        "42.5"
        >>> extract_numerical_answer("Final answer: 123.456")
        "123.456"
    """
    if not response:
        return None
    
    # Look for common answer patterns (ordered by specificity)
    patterns = [
        # "Therefore, $5,542.86 was invested" - most specific
        r"therefore,?\s*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*was invested",
        # "Therefore, 8.78 hours" - time answers
        r"therefore,?.*?([+-]?\d+(?:\.\d+)?)\s*hours?",
        # "The faster train will overtake... 8.78 hours after"
        r"overtake.*?([+-]?\d+(?:\.\d+)?)\s*hours?\s*after",
        # "$5,542.86 was invested at" - investment answers
        r"\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*was invested",
        # "The answer is 42.5" or "answer = 42.5"
        r"(?:answer is|equals?|=)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        # "Final answer: 123.456"
        r"final answer:?\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        # Last resort: "Therefore, X" (any number after therefore)
        r"therefore,?\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            # Remove commas and return clean number
            return match.group(1).replace(',', '').lstrip('+')

    # NO FALLBACK - for maximum experimental rigor
    return None


def validate_api_key(api_key: str, service_name: str) -> bool:
    """
    Validate that an API key looks reasonable.
    
    Args:
        api_key: The API key to validate
        service_name: Name of the service for error messages
        
    Returns:
        True if key looks valid, False otherwise
    """
    if not api_key:
        return False
    
    if api_key in ['your_api_key_here', 'your_key_here']:
        return False
    
    if api_key.startswith('your_'):
        return False
    
    # Basic format checks
    if service_name.lower() == 'openai' and not api_key.startswith('sk-'):
        return False
    
    if service_name.lower() == 'anthropic' and not api_key.startswith('sk-ant-'):
        return False
    
    return True


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
        
    Examples:
        >>> format_duration(65.5)
        "1m 5.5s"
        >>> format_duration(3661.2)
        "1h 1m 1.2s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.1f}s"
    
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    
    return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"
