import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression

class FeedbackLearner:
    """
    Adaptive Machine Learning Feedback Layer:
    Learns dynamic feature weights (w1: Text, w2: MBTI, w3: Location)
    from user interaction history (Accept = 1, Reject = 0).
    """

    DEFAULT_WEIGHTS = {"w1_text": 0.50, "w2_mbti": 0.30, "w3_location": 0.20}

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.data_dir = data_dir
        self.user_weights_cache = {}
        self.global_weights = dict(self.DEFAULT_WEIGHTS)
        self.feedback_df = None
        
    def load_feedback_data(self):
        fb_path = os.path.join(self.data_dir, "feedback.csv")
        if os.path.exists(fb_path):
            self.feedback_df = pd.read_csv(fb_path)
            return self.feedback_df
        return None

    def learn_weights_from_interactions(self, interaction_features_df: pd.DataFrame):
        """
        Trains model on historical interaction features [text_sim, mbti_match, location_sim] -> action.
        Extracts global & per-user weights.
        """
        if interaction_features_df is None or len(interaction_features_df) < 5:
            return self.global_weights

        X = interaction_features_df[["text_sim", "mbti_match", "location_sim"]].values
        y = interaction_features_df["action"].values

        # Global learning via Ridge regression (stable non-negative weights)
        model = Ridge(alpha=1.0, positive=True, fit_intercept=False)
        model.fit(X, y)
        coefs = model.coef_

        if coefs.sum() > 0:
            norm_coefs = coefs / coefs.sum()
            self.global_weights = {
                "w1_text": float(round(norm_coefs[0], 3)),
                "w2_mbti": float(round(norm_coefs[1], 3)),
                "w3_location": float(round(norm_coefs[2], 3))
            }

        # Learn per-user weights if user has at least 4 interactions
        for user_id, group in interaction_features_df.groupby("user_id"):
            if len(group) >= 4 and len(group["action"].unique()) > 1:
                u_X = group[["text_sim", "mbti_match", "location_sim"]].values
                u_y = group["action"].values
                u_model = Ridge(alpha=0.5, positive=True, fit_intercept=False)
                u_model.fit(u_X, u_y)
                u_coefs = u_model.coef_
                if u_coefs.sum() > 0:
                    u_norm = u_coefs / u_coefs.sum()
                    self.user_weights_cache[user_id] = {
                        "w1_text": float(round(u_norm[0], 3)),
                        "w2_mbti": float(round(u_norm[1], 3)),
                        "w3_location": float(round(u_norm[2], 3))
                    }

        return self.global_weights

    def get_user_weights(self, user_id: str) -> dict:
        """
        Returns personalized weights for a given user, or global weights if no custom model exists.
        """
        return self.user_weights_cache.get(user_id, self.global_weights)

    def update_single_feedback(self, user_id: str, text_sim: float, mbti_match: float, location_sim: float, action: int):
        """
        Real-time online weight adaptation when a user clicks Accept (1) or Reject (0) in the UI.
        Uses Gradient Descent update on user's weight vector.
        """
        current_w = self.get_user_weights(user_id)
        w_vec = np.array([current_w["w1_text"], current_w["w2_mbti"], current_w["w3_location"]])
        x_vec = np.array([text_sim, mbti_match, location_sim])

        pred = np.dot(w_vec, x_vec)
        error = action - pred
        lr = 0.15

        # Gradient update
        new_w_vec = w_vec + lr * error * x_vec
        new_w_vec = np.maximum(new_w_vec, 0.05) # Keep strictly positive
        new_w_vec = new_w_vec / new_w_vec.sum() # Normalize

        updated_dict = {
            "w1_text": float(round(new_w_vec[0], 3)),
            "w2_mbti": float(round(new_w_vec[1], 3)),
            "w3_location": float(round(new_w_vec[2], 3))
        }
        self.user_weights_cache[user_id] = updated_dict
        return updated_dict
