#!/usr/bin/env python3
"""
Debug the answer extraction issue in Phase 2 results.
"""

import re
from typing import Optional

def extract_numerical_answer_debug(response: str) -> Optional[str]:
    """Debug version of extract_numerical_answer with detailed logging."""
    if not response:
        return None
    
    patterns = [
        (r'[tx]\s*[≈=]\s*([+-]?\d+(?:\.\d+)?)', 'Mathematical expressions (t = X)'),
        (r'≈\s*([+-]?\d+(?:\.\d+)?)\s*(?:hours?|years?|units?)', 'Time with ≈'),
        (r'([+-]?\d+(?:\.\d+)?)\s*hours?\s*(?:after|at)', 'Hours after/at'),
        (r'([+-]?\d+(?:\.\d+)?)\s*hours?(?:\s*$|\s*[^\d])', 'Hours general'),
        (r'speed.*?is\s*([+-]?\d+(?:\.\d+)?)\s*mph', 'Speed expressions'),
        (r'is\s*([+-]?\d+(?:\.\d+)?)\s*mph', 'Is X mph'),
        (r'therefore,?\s*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*was invested', 'Investment therefore'),
        (r'\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*was invested', 'Investment general'),
        (r'therefore,?\s*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', 'Therefore money'),
        (r'therefore,?.*?([+-]?\d+(?:\.\d+)?)\s*hours?', 'Therefore hours'),
        (r'overtake.*?([+-]?\d+(?:\.\d+)?)\s*hours?\s*after', 'Overtake hours'),
        (r'(?:answer is|equals?|=)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', 'Answer is'),
        (r'final answer:?\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', 'Final answer'),
        (r'([+-]?\d+(?:\.\d+)?)\s*$', 'End number'),
    ]
    
    print("DEBUGGING EXTRACTION:")
    print("=" * 50)
    print(f"Response length: {len(response)}")
    print(f"Last 200 chars: ...{response[-200:]}")
    print()
    
    for i, (pattern, desc) in enumerate(patterns):
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            print(f"✅ Pattern {i+1} ({desc}):")
            print(f"   Regex: {pattern}")
            for j, match in enumerate(matches):
                print(f"   Match {j+1}: \"{match.group(0)}\" -> {match.group(1)} (pos {match.start()}-{match.end()})")
            
            # Return the first match
            result = matches[0].group(1).replace(',', '').lstrip('+')
            print(f"   SELECTED: {result}")
            return result
        else:
            print(f"❌ Pattern {i+1} ({desc}): No match")
    
    return None


def test_with_claude_response():
    """Test with the actual Claude response that's being misextracted."""
    
    claude_response = """I need to find when the faster train catches up to the slower train.

Let me define variables:
- Let t = time (in hours) after the faster train departs when it overtakes the slower train

When the faster train overtakes the slower train, both trains will have traveled the same distance from the station.

Distance traveled by slower train:
- It had a 2-hour head start, so it travels for (t + 2) hours total
- Distance = 72(t + 2) km

Distance traveled by faster train:
- It travels for t hours
- Distance = 88.4t km

Setting the distances equal:
72(t + 2) = 88.4t

Expanding:
72t + 144 = 88.4t

Solving for t:
144 = 88.4t - 72t
144 = 16.4t
t = 144/16.4
t = 8.78 hours

Converting to hours and minutes:
0.78 hours × 60 minutes/hour ≈ 47 minutes

Therefore, the faster train will overtake the slower train approximately 8 hours and 47 minutes after the faster train departs."""

    print("TESTING WITH CLAUDE RESPONSE:")
    print("=" * 60)
    
    result = extract_numerical_answer_debug(claude_response)
    print(f"\nFINAL EXTRACTED: {result}")
    print(f"EXPECTED: 8.78048780487805")
    print(f"CORRECT? {'✅ YES' if result and abs(float(result) - 8.78048780487805) < 0.01 else '❌ NO'}")


if __name__ == "__main__":
    test_with_claude_response()
