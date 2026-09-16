import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from src.matching_engine import MatchingEngine

# Page Configuration
st.set_page_config(
    page_title="Intelligent Profile Matching Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .profile-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #2a5298;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .match-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.25rem;
        transition: transform 0.2s ease;
    }
    .mbti-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
    }
    .score-badge {
        font-size: 1.6rem;
        font-weight: 800;
        color: #2b6cb0;
    }
    .weight-chip {
        background-color: #edf2f7;
        color: #2d3748;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Data Directory initialization
PROJECT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

@st.cache_resource
def load_engine():
    return MatchingEngine(data_dir=DATA_DIR)

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error loading matching engine: {e}")
    st.stop()

# Sidebar: User Selection & Global Controls
st.sidebar.markdown("### 👤 User Persona Selector")
user_ids = list(engine.users_dict.keys())
selected_user_id = st.sidebar.selectbox(
    "Select Target User:",
    user_ids,
    format_func=lambda uid: f"{uid} - {engine.users_dict[uid]['name']} ({engine.users_dict[uid]['profession']})"
)

current_user = engine.users_dict[selected_user_id]
user_weights = engine.feedback_learner.get_user_weights(selected_user_id)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Current Active Weights")
st.sidebar.markdown(f"""
- **Text Similarity ($w_1$)**: `{user_weights['w1_text']:.2f}`
- **MBTI Match ($w_2$)**: `{user_weights['w2_mbti']:.2f}`
- **Location ($w_3$)**: `{user_weights['w3_location']:.2f}`
""")

if st.sidebar.button("🔄 Reset User Weights"):
    engine.feedback_learner.user_weights_cache[selected_user_id] = dict(engine.feedback_learner.DEFAULT_WEIGHTS)
    st.rerun()

# Main Layout Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Top Matches & Recommendation", "🔬 Custom Pair Matcher", "📈 Feedback Analytics"])

# TAB 1: TOP MATCHES & RECOMMENDATION ENGINE
with tab1:
    st.markdown('<div class="main-title">Intelligent Matching System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hybrid recommendation synthesizing NLP semantics, MBTI cognitive matrix, and real-time ML feedback</div>', unsafe_allow_html=True)

    # Current User Profile Header Card
    with st.container():
        st.markdown(f"""
        <div class="profile-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2>{current_user['name']} <span class="mbti-badge">{current_user['mbti']}</span></h2>
                <div>📍 {current_user['location']} | 💼 {current_user['profession']} ({current_user['experience_years']} yrs exp)</div>
            </div>
            <p><strong>Professional Summary:</strong> {current_user['professional_summary']}</p>
            <p><strong>About Me:</strong> {current_user['about_me']}</p>
            <p><strong>Interests:</strong> {current_user['interests']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🌟 Top 5 Recommended Matches")

    # Get Top 5 Matches
    top_matches = engine.get_top_matches(selected_user_id, k=5)

    for idx, match in enumerate(top_matches):
        p = match["matched_profile"]
        comps = match["components"]
        score = match["compatibility_score"]
        m_id = p["user_id"]

        with st.container():
            col1, col2, col3 = st.columns([3, 4, 2])
            
            with col1:
                st.markdown(f"#### #{idx+1}. {p['name']} <span class='mbti-badge'>{p['mbti']}</span>", unsafe_allow_html=True)
                st.markdown(f"**{p['profession']}** ({p['experience_years']} yrs) | 📍 {p['location']}")
                st.caption(f"**Interests:** {p['interests']}")

            with col2:
                st.markdown(f"**Compatibility Score:** <span class='score-badge'>{score}%</span>", unsafe_allow_html=True)
                st.progress(int(score))
                st.caption(f"📝 Text Sim: **{comps['text_similarity']}%** | 🧩 MBTI: **{comps['mbti_match']}%** | 📍 Location: **{comps['location_similarity']}%**")

            with col3:
                st.write("")
                st.write("**Feedback Actions:**")
                btn_col1, btn_col2 = st.columns(2)
                
                if btn_col1.button("👍 Accept", key=f"acc_{selected_user_id}_{m_id}_{idx}"):
                    # Retrieve component features 0..1 scale
                    feats = engine.calculate_pair_features(current_user, p)
                    new_w = engine.feedback_learner.update_single_feedback(
                        selected_user_id, feats["text_sim"], feats["mbti_match"], feats["location_sim"], action=1
                    )
                    st.toast(f"✅ Match Accepted! Updated weights: w1={new_w['w1_text']:.2f}, w2={new_w['w2_mbti']:.2f}", icon="🎉")
                    st.rerun()

                if btn_col2.button("👎 Reject", key=f"rej_{selected_user_id}_{m_id}_{idx}"):
                    feats = engine.calculate_pair_features(current_user, p)
                    new_w = engine.feedback_learner.update_single_feedback(
                        selected_user_id, feats["text_sim"], feats["mbti_match"], feats["location_sim"], action=0
                    )
                    st.toast(f"❌ Match Rejected! Model re-trained for {selected_user_id}", icon="⚡")
                    st.rerun()

            with st.expander(f"View Full Profile Details for {p['name']}"):
                st.write(f"**Professional Summary:** {p['professional_summary']}")
                st.write(f"**About Me:** {p['about_me']}")

            st.markdown("---")

# TAB 2: CUSTOM PAIR MATCHER
with tab2:
    st.markdown("### 🔬 Compare Any Two Profiles")
    c1, c2 = st.columns(2)
    with c1:
        u1_id = st.selectbox("Select First User:", user_ids, index=0, key="pair_u1")
    with c2:
        u2_id = st.selectbox("Select Second User:", user_ids, index=min(1, len(user_ids)-1), key="pair_u2")

    if u1_id == u2_id:
        st.warning("Please select two distinct users to compute compatibility.")
    else:
        res = engine.calculate_compatibility(u1_id, u2_id)
        st.markdown(f"## Overall Compatibility: **{res['compatibility_score']}%**")
        
        comp = res["components"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("NLP Text Similarity", f"{comp['text_similarity']}%")
        col_b.metric("MBTI Match Score", f"{comp['mbti_match']}%")
        col_c.metric("Location Match", f"{comp['location_similarity']}%")

        st.markdown("#### Formula Breakdown")
        w = res["weights_used"]
        st.latex(rf"\text{{Score}} = ({w['w1_text']:.2f} \times {comp['text_similarity']}) + ({w['w2_mbti']:.2f} \times {comp['mbti_match']}) + ({w['w3_location']:.2f} \times {comp['location_similarity']}) = {res['compatibility_score']}\%")

# TAB 3: FEEDBACK ANALYTICS & LEARNING
with tab3:
    st.markdown("### 📈 Machine Learning Feedback Loop Performance")
    st.markdown("This tab displays evaluation metrics demonstrating how interactive Accept/Reject learning improves matching precision over baseline static logic.")

    eval_plot_path = os.path.join(DATA_DIR, "feedback_performance_analysis.png")
    if os.path.exists(eval_plot_path):
        st.image(eval_plot_path, use_container_width=True)
    else:
        st.info("Run `python src/evaluate.py` to generate evaluation plots.")

    eval_report_path = os.path.join(DATA_DIR, "evaluation_report.md")
    if os.path.exists(eval_report_path):
        with open(eval_report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
