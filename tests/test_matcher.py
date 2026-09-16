import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from src.nlp_engine import NLPEngine
from src.mbti_matrix import MBTIMatcher
from src.matching_engine import MatchingEngine
from src.feedback_learner import FeedbackLearner

def test_nlp_engine():
    engine = NLPEngine()
    text1 = "Data Analyst skilled in Python, SQL, and machine learning."
    text2 = "Senior Data Scientist proficient in Python, statistics, and machine learning."
    text3 = "Professional chef with 5 years in French pastry and baking."

    sim1_2 = engine.compute_similarity(text1, text2)
    sim1_3 = engine.compute_similarity(text1, text3)

    assert sim1_2 > sim1_3, "Data analyst & scientist text should have higher similarity than chef text."
    assert 0.0 <= sim1_2 <= 1.0

def test_mbti_matcher():
    # Test dual ideal pair
    score_ideal = MBTIMatcher.get_compatibility_score("INTJ", "ENFP")
    assert score_ideal == 1.0, f"INTJ-ENFP should be 1.0, got {score_ideal}"

    # Test same type
    score_same = MBTIMatcher.get_compatibility_score("INTP", "INTP")
    assert score_same == 0.75

    # Test conflicting/friction types
    score_conflict = MBTIMatcher.get_compatibility_score("INTJ", "ESFP")
    assert score_conflict < 0.70

def test_matching_engine():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_dir, "data")
    engine = MatchingEngine(data_dir=data_dir)

    # Top matches for U001
    top_matches = engine.get_top_matches("U001", k=5)
    assert len(top_matches) == 5
    assert top_matches[0]["compatibility_score"] >= top_matches[1]["compatibility_score"]

    for match in top_matches:
        score = match["compatibility_score"]
        assert 0.0 <= score <= 100.0
        assert "components" in match
        assert "text_similarity" in match["components"]
        assert "mbti_match" in match["components"]

def test_feedback_learner():
    learner = FeedbackLearner()
    initial_w = learner.get_user_weights("U001")
    assert "w1_text" in initial_w

    # User clicks Accept (1) on a candidate with high text sim (0.9) and low MBTI (0.2)
    updated_w = learner.update_single_feedback(
        user_id="U001",
        text_sim=0.9,
        mbti_match=0.2,
        location_sim=0.1,
        action=1
    )

    # w1_text should increase because user accepted a match driven by high text similarity
    assert updated_w["w1_text"] > initial_w["w1_text"]
