import os
import pandas as pd
import numpy as np
from src.nlp_engine import NLPEngine
from src.mbti_matrix import MBTIMatcher
from src.feedback_learner import FeedbackLearner

class MatchingEngine:
    """
    Profile Scoring Engine (The "Logic" & "Hybrid" Layer):
    Combines NLP text similarity, MBTI personality logic, and Location match
    into a unified Compatibility Score (0 - 100%).
    Integrated with adaptive ML feedback learner.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.data_dir = data_dir
        self.nlp_engine = NLPEngine()
        self.feedback_learner = FeedbackLearner(data_dir=data_dir)
        self.users_df = None
        self.users_dict = {}
        
        self.load_data()
        self.initialize_nlp_corpus()
        self.train_feedback_weights()

    def load_data(self):
        users_path = os.path.join(self.data_dir, "users.csv")
        if os.path.exists(users_path):
            self.users_df = pd.read_csv(users_path)
            # Store in dictionary for quick lookup by user_id
            self.users_dict = {
                row["user_id"]: row.to_dict() for _, row in self.users_df.iterrows()
            }
        else:
            raise FileNotFoundError(f"users.csv not found at {users_path}")

    def initialize_nlp_corpus(self):
        """
        Fits TF-IDF vectorizer on combined professional summaries & about me fields.
        """
        combined_texts = []
        for u_id, u_data in self.users_dict.items():
            text = f"{u_data.get('professional_summary', '')} {u_data.get('about_me', '')} {u_data.get('interests', '')}"
            combined_texts.append(text)
        self.nlp_engine.fit_transform_corpus(combined_texts)

    def calculate_pair_features(self, u1_data: dict, u2_data: dict) -> dict:
        """
        Calculates individual component similarities [TextSim, MBTIMatch, LocationSim] between two user profiles.
        """
        # 1. NLP Text Similarity (Professional Summary + About Me + Interests)
        t1 = f"{u1_data.get('professional_summary', '')} {u1_data.get('about_me', '')} {u1_data.get('interests', '')}"
        t2 = f"{u2_data.get('professional_summary', '')} {u2_data.get('about_me', '')} {u2_data.get('interests', '')}"
        text_sim = self.nlp_engine.compute_similarity(t1, t2)

        # 2. MBTI Personality Compatibility
        mbti_match = MBTIMatcher.get_compatibility_score(
            u1_data.get("mbti", ""), u2_data.get("mbti", "")
        )

        # 3. Demographic / Location Similarity
        loc1 = str(u1_data.get("location", "")).lower().strip()
        loc2 = str(u2_data.get("location", "")).lower().strip()
        if loc1 == loc2:
            location_sim = 1.0
        elif loc1.split(",")[-1].strip() == loc2.split(",")[-1].strip(): # Same country
            location_sim = 0.5
        else:
            location_sim = 0.1

        return {
            "text_sim": float(round(text_sim, 4)),
            "mbti_match": float(round(mbti_match, 4)),
            "location_sim": float(round(location_sim, 4))
        }

    def train_feedback_weights(self):
        """
        Loads feedback dataset, computes interaction feature vectors, and trains feedback weights.
        """
        fb_df = self.feedback_learner.load_feedback_data()
        if fb_df is None or len(fb_df) == 0:
            return

        features_list = []
        for _, row in fb_df.iterrows():
            u1, u2, act = str(row["user_id"]), str(row["matched_user_id"]), int(row["action"])
            if u1 in self.users_dict and u2 in self.users_dict:
                feats = self.calculate_pair_features(self.users_dict[u1], self.users_dict[u2])
                feats["user_id"] = u1
                feats["matched_user_id"] = u2
                feats["action"] = act
                features_list.append(feats)

        if features_list:
            feat_df = pd.DataFrame(features_list)
            self.feedback_learner.learn_weights_from_interactions(feat_df)

    def calculate_compatibility(self, user_id_1: str, user_id_2: str, custom_weights: dict = None) -> dict:
        """
        Calculates hybrid compatibility score between User 1 and User 2.
        Core Formula: TotalScore = (w1 * TextSim) + (w2 * MBTIMatch) + (w3 * LocationSim)
        """
        if user_id_1 not in self.users_dict or user_id_2 not in self.users_dict:
            raise ValueError(f"One or both User IDs ({user_id_1}, {user_id_2}) not found.")

        u1_data = self.users_dict[user_id_1]
        u2_data = self.users_dict[user_id_2]

        feats = self.calculate_pair_features(u1_data, u2_data)

        # Get weights
        weights = custom_weights if custom_weights else self.feedback_learner.get_user_weights(user_id_1)
        w1, w2, w3 = weights["w1_text"], weights["w2_mbti"], weights["w3_location"]

        total_score_raw = (w1 * feats["text_sim"]) + (w2 * feats["mbti_match"]) + (w3 * feats["location_sim"])
        # Scale to percentage 0 - 100%
        compatibility_score_pct = round(min(100.0, max(0.0, total_score_raw * 100.0)), 2)

        return {
            "user_id": user_id_1,
            "matched_user_id": user_id_2,
            "compatibility_score": compatibility_score_pct,
            "components": {
                "text_similarity": round(feats["text_sim"] * 100.0, 1),
                "mbti_match": round(feats["mbti_match"] * 100.0, 1),
                "location_similarity": round(feats["location_sim"] * 100.0, 1)
            },
            "weights_used": weights,
            "matched_profile": u2_data
        }

    def get_top_matches(self, user_id: str, k: int = 5) -> list:
        """
        Ranks all potential user profiles for a given target user and returns Top K matches.
        """
        if user_id not in self.users_dict:
            raise ValueError(f"User ID {user_id} not found.")

        results = []
        for other_id in self.users_dict:
            if other_id == user_id:
                continue
            res = self.calculate_compatibility(user_id, other_id)
            results.append(res)

        # Sort descending by compatibility_score
        results.sort(key=lambda x: x["compatibility_score"], reverse=True)
        return results[:k]
