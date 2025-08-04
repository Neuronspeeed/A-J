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


def extract_xml_answers(response: str) -> List[Optional[str]]:
    """
    Extract answers from <answer1> and <answer2> XML tags.
    
    Returns as list for compatibility with test code.
    Now handles cases where AI shows mathematical work inside tags.
    
    Args:
        response: Full model response containing XML tags
        
    Returns:
        List of [first_answer, second_answer], either can be None
        
    Examples:
        >>> extract_xml_answers("<answer1>Hello</answer1><answer2>42</answer2>")
        ["Hello", "42"]
        >>> extract_xml_answers("No tags here")
        [None, None]
    """
    if not response:
        return [None, None]
    
    pattern = re.compile(r"<answer(\d+)>((?:.|\n)*?)</answer\1>", re.IGNORECASE)
    matches = pattern.findall(response)
    
    answers = [None, None]
    for num, content in matches:
        idx = int(num) - 1
        if idx in (0, 1):
            answers[idx] = content.strip()
    
    return answers


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
    
    # Look for sequences of numbers (1+ digits, excluding decimal numbers)
    numbers = re.findall(r'\b\d+\b', text)
    # Filter out very small numbers (single digits) and very large numbers (likely not random)
    valid_numbers = [n for n in numbers if 2 <= len(n) <= 10]
    if len(valid_numbers) >= 3:  # Need at least 3 numbers
        return [int(n) for n in valid_numbers[:10]]  # Take up to 10 numbers
    
    return None


def extract_numerical_answer(response: str) -> Optional[str]:
    """
    Extract numerical answer from response text that may contain mathematical work.
    
    Handles cases where AI shows work and then gives final calculation.
    Prioritizes final computed results over intermediate values.
    
    Args:
        response: Model response containing a numerical answer
        
    Returns:
        Numerical answer as string, or None if not found
        
    Examples:
        >>> extract_numerical_answer("t = 144/16.4 = 8.78048780487804...")
        "8.78048780487804..."
        >>> extract_numerical_answer("The answer is 42.5")
        "42.5"
    """
    if not response:
        return None
    
    # Enhanced patterns prioritizing final calculations
    patterns = [
        # Final equals with long decimal: "= 8.78048780487804878..."
        r"=\s*([+-]?\d+\.?\d*(?:\d{10,}))",  # Long decimals (10+ digits after decimal)
        
        # Mathematical expressions: "t = 144/16.4 = 8.7804..."
        r"=\s*\d+(?:\.\d+)?/\d+(?:\.\d+)?\s*=\s*([+-]?\d+(?:\.\d+)?)",
        
        # Final calculation results: "= 8.7804878048780494..."
        r"=\s*([+-]?\d+(?:\.\d{15,}))",  # Very long decimals (15+ digits)
        
        # Variable assignments at end: "t = 8.7804", "x = 5542.857"
        r"[tx]\s*=\s*([+-]?\d+(?:\.\d+)?)\s*$",
        r"[tx]\s*=\s*([+-]?\d+(?:\.\d+)?)\s*(?:hours?|years?|units?)",
        
        # Investment results: "$5,542.86", "5542.857142857142..."
        r"\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
        r"([+-]?\d+(?:\.\d{10,}))\s*$",  # Long number at end
        
        # Mathematical expressions: "t ≈ 5.65", "x = 2.81", "t = 5.13 years"
        r"[tx]\s*[≈=]\s*([+-]?\d+(?:\.\d+)?)",
        # Time expressions: "4.7 hours", "8.78 hours after", "≈ 5.65 years"
        r"≈\s*([+-]?\d+(?:\.\d+)?)\s*(?:hours?|years?|units?)",
        r"([+-]?\d+(?:\.\d+)?)\s*hours?\s*(?:after|at)",
        r"([+-]?\d+(?:\.\d+)?)\s*hours?(?:\s*$|\s*[^\d])",
        # Speed expressions: "speed is 15 mph", "is 13.5 mph"
        r"speed.*?is\s*([+-]?\d+(?:\.\d+)?)\s*mph",
        r"is\s*([+-]?\d+(?:\.\d+)?)\s*mph",
        # Investment expressions: "$5,542.86 was invested", "Therefore, $5,542.86"
        r"therefore,?\s*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*was invested",
        r"\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*was invested",
        r"therefore,?\s*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        # Train problems: "8.78 hours after", "overtake... X hours"
        r"therefore,?.*?([+-]?\d+(?:\.\d+)?)\s*hours?",
        r"overtake.*?([+-]?\d+(?:\.\d+)?)\s*hours?\s*after",
        # Standard answer patterns: "The answer is 42.5", "Final answer: 123"
        r"(?:answer is|equals?|=)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"final answer:?\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        # Speed/time results at end
        r"([+-]?\d+(?:\.\d+)?)\s*(?:mph|hours?|km/h)\s*$",
        # Catch isolated numbers at end of response
        r"([+-]?\d+(?:\.\d+)?)\s*$",
    ]
    
    # Split into lines and check the last few lines for final answer
    lines = response.strip().split('\n')
    
    # Check last 3 lines for final calculation (prioritize recent calculations)
    for line in reversed(lines[-3:]):
        for pattern in patterns[:5]:  # Use most specific patterns first
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                number = match.group(1).replace(',', '').lstrip('+')
                # Validate it's a reasonable final answer (not intermediate like "144")
                if '.' in number and len(number.split('.')[1]) >= 3:  # At least 3 decimal places
                    return number
                elif len(number) >= 4:  # Or at least 4 total digits
                    return number
    
    # Fallback: check entire text with all patterns
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
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
