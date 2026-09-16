"""
MBTI Personality Logic & Compatibility Matrix Layer
Provides static rule dictionary and scoring matrix for 16 Myers-Briggs types.
"""

# List of all 16 valid MBTI personality types
VALID_MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"
]

class MBTIMatcher:
    """
    Computes compatibility score between any two MBTI types (0.0 to 1.0).
    """

    # Ideal / High Compatibility Pairings Matrix Dictionary
    # Dual and complementary function pairs get highest scores (1.0), opposing/friction types lower.
    PREDEFINED_SCORES = {
        # Analysts (NT)
        ("INTJ", "ENFP"): 1.00, ("INTJ", "ENTP"): 0.95, ("INTJ", "ENTJ"): 0.85, ("INTJ", "INTP"): 0.80,
        ("INTP", "ENTJ"): 1.00, ("INTP", "ENFJ"): 0.90, ("INTP", "INTJ"): 0.80, ("INTP", "ENTP"): 0.85,
        ("ENTJ", "INTP"): 1.00, ("ENTJ", "INFP"): 0.95, ("ENTJ", "INTJ"): 0.85, ("ENTJ", "ENFJ"): 0.80,
        ("ENTP", "INFJ"): 1.00, ("ENTP", "INTJ"): 0.95, ("ENTP", "ENFP"): 0.85, ("ENTP", "INTP"): 0.85,

        # Diplomats (NF)
        ("INFJ", "ENTP"): 1.00, ("INFJ", "ENFP"): 0.95, ("INFJ", "ENFJ"): 0.85, ("INFJ", "INFP"): 0.80,
        ("INFP", "ENFJ"): 1.00, ("INFP", "ENTJ"): 0.95, ("INFP", "INFJ"): 0.80, ("INFP", "ENFP"): 0.85,
        ("ENFJ", "INFP"): 1.00, ("ENFJ", "INTP"): 0.90, ("ENFJ", "INFJ"): 0.85, ("ENFJ", "ENFP"): 0.80,
        ("ENFP", "INTJ"): 1.00, ("ENFP", "INFJ"): 0.95, ("ENFP", "ENTP"): 0.85, ("ENFP", "INFP"): 0.85,

        # Sentinels (SJ)
        ("ISTJ", "ESFP"): 1.00, ("ISTJ", "ESTP"): 0.90, ("ISTJ", "ESTJ"): 0.85, ("ISTJ", "ISFJ"): 0.80,
        ("ISFJ", "ESTP"): 1.00, ("ISFJ", "ESFP"): 0.90, ("ISFJ", "ESFJ"): 0.85, ("ISFJ", "ISTJ"): 0.80,
        ("ESTJ", "ISFP"): 1.00, ("ESTJ", "ISTP"): 0.90, ("ESTJ", "ISTJ"): 0.85, ("ESTJ", "ESFJ"): 0.80,
        ("ESFJ", "ISTP"): 1.00, ("ESFJ", "ISFP"): 0.90, ("ESFJ", "ISFJ"): 0.85, ("ESFJ", "ESTJ"): 0.80,

        # Explorers (SP)
        ("ISTP", "ESFJ"): 1.00, ("ISTP", "ESTJ"): 0.90, ("ISTP", "ESTP"): 0.85, ("ISTP", "ISFP"): 0.80,
        ("ISFP", "ESTJ"): 1.00, ("ISFP", "ESFJ"): 0.90, ("ISFP", "ESFP"): 0.85, ("ISFP", "ISTP"): 0.80,
        ("ESTP", "ISFJ"): 1.00, ("ESTP", "ISTJ"): 0.90, ("ESTP", "ISTP"): 0.85, ("ESTP", "ESFP"): 0.80,
        ("ESFP", "ISTJ"): 1.00, ("ESFP", "ISFJ"): 0.90, ("ESFP", "ISFP"): 0.85, ("ESFP", "ESTP"): 0.80,
    }

    @classmethod
    def get_compatibility_score(cls, mbti1: str, mbti2: str) -> float:
        """
        Returns compatibility score in [0.0, 1.0] for two MBTI strings.
        Uses predefined matrix or falls back to dimensional function overlap calculation.
        """
        if not mbti1 or not mbti2:
            return 0.5
        
        m1, m2 = mbti1.upper().strip(), mbti2.upper().strip()
        if m1 not in VALID_MBTI_TYPES or m2 not in VALID_MBTI_TYPES:
            return 0.5
        
        # Check direct lookup (both directions)
        if (m1, m2) in cls.PREDEFINED_SCORES:
            return cls.PREDEFINED_SCORES[(m1, m2)]
        if (m2, m1) in cls.PREDEFINED_SCORES:
            return cls.PREDEFINED_SCORES[(m2, m1)]

        # Same MBTI type
        if m1 == m2:
            return 0.75

        # Fallback: Dimension-wise cognitive function overlap
        # Compare E/I, N/S, T/F, J/P
        matches = sum(1 for a, b in zip(m1, m2) if a == b)
        
        # N/S dimension is crucial for communication style
        ns_match = (m1[1] == m2[1])
        
        base_score = 0.40 + (matches * 0.10)
        if ns_match:
            base_score += 0.10

        return round(min(1.0, base_score), 2)
