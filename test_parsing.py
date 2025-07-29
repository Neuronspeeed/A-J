#!/usr/bin/env python3
"""Test the improved numerical answer extraction."""

from core.utils import extract_numerical_answer

# Test cases from actual baseline responses
test_cases = [
    ("Therefore, $5,542.86 was invested at 6.7%.", "5542.86"),
    ("The faster train will overtake the slower train 8.78 hours after it departs.", "8.78"),
    ("Therefore, $5,542.86 was invested at 6.7%.", "5542.86"),
    ("The answer is 42.5", "42.5"),
    ("Final answer: 123.456", "123.456"),
    ("It equals 99.9 dollars", "99.9"),
    # Test problematic case that was extracting 88.4 instead of 8.78
    ("A train at 72 km/h, then 88.4 km/h. The faster train will overtake 8.78 hours after.", "8.78"),
]

print("TESTING IMPROVED NUMERICAL EXTRACTION")
print("=" * 50)

for i, (text, expected) in enumerate(test_cases, 1):
    result = extract_numerical_answer(text)
    status = "✅" if result == expected else "❌"
    print(f"Test {i}: {status}")
    print(f"  Input: {text}")
    print(f"  Expected: {expected}")
    print(f"  Got: {result}")
    print()

print("Testing with actual baseline response:")
baseline_response = """I need to find how much was invested at 6.7% interest rate.

Let me define variables:
- Let x = amount invested at 6.7%
- Then (12,500 - x) = amount invested at 5.3%

The total interest equation is:
0.067x + 0.053(12,500 - x) = 740.10

Expanding:
0.067x + 662.5 - 0.053x = 740.10

Combining like terms:
0.014x + 662.5 = 740.10

Solving for x:
0.014x = 740.10 - 662.5
0.014x = 77.60
x = 77.60 ÷ 0.014
x = 5,542.857...
x ≈ 5,542.86

Let me verify:
- Amount at 6.7%: $5,542.86
- Amount at 5.3%: $12,500 - $5,542.86 = $6,957.14
- Interest from 6.7%: $5,542.86 × 0.067 = $371.37
- Interest from 5.3%: $6,957.14 × 0.053 = $368.73
- Total interest: $371.37 + $368.73 = $740.10 ✓

Therefore, $5,542.86 was invested at 6.7%."""

result = extract_numerical_answer(baseline_response)
print(f"Baseline response result: {result}")
print(f"Expected: 5542.86")
print(f"Match: {'✅' if result == '5542.86' else '❌'}")
