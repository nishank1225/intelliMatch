# Intelligent Profile-Based Matching System

A self-learning Python recommendation system that calculates holistic compatibility scores between users combining **Unstructured Text (NLP)**, **Myers-Briggs Personality Logic (MBTI)**, and **Demographics**, enhanced with an **Adaptive Machine Learning Feedback Loop** that updates scoring parameters based on real-time user actions (Accept/Reject).

---

## 🌟 Key Features

1. **Unstructured Data Parsing (NLP Layer)**:
   - Cleans raw bios and professional summaries via text normalization, stopword removal, and rule-based lemmatization.
   - Converts unstructured free-text fields into numerical vector representations using TF-IDF.
   - Measures semantic similarity using Cosine Distance.

2. **Hybrid Scoring Engine (Logic Layer)**:
   - Evaluates Myers-Briggs personality pairings using a scientifically grounded 16x16 MBTI cognitive compatibility matrix (Dual, complementary, and friction pairings).
   - Incorporates location and demographic proximity.
   - Synthesizes features into a unified score ($0 - 100\%$):
     $$\text{TotalScore} = (w_1 \times \text{TextSim}) + (w_2 \times \text{MBTIMatch}) + (w_3 \times \text{LocationSim})$$

3. **Adaptive Feedback Loop (ML Layer)**:
   - Logs user Accept (1) and Reject (0) interactions.
   - Dynamically learns per-user or global feature weights ($w_1, w_2, w_3$) using Ridge regression / online gradient updates.
   - Adjusts weights automatically (e.g., if a user values professional text similarity over MBTI, $w_1$ increases while $w_2$ decreases).

4. **Interactive Streamlit Web Dashboard**:
   - Live web application showcasing profile selection, top 5 candidate recommendations with detailed percentage breakdowns, interactive Accept/Reject feedback buttons, real-time weight re-calculation, and performance analytics.

---

## 📁 Repository Structure

```
ML_Project/
├── data/
│   ├── users.csv                     # Synthetic user profiles (100 profiles)
│   ├── feedback.csv                  # User interaction history (800+ rows)
│   ├── feedback_performance_analysis.png # Performance evaluation chart
│   └── evaluation_report.md          # Metrics summary report
├── src/
│   ├── __init__.py
│   ├── dataset_generator.py          # Data pipeline script generating realistic synthetic data
│   ├── nlp_engine.py                 # Text preprocessing & TF-IDF Cosine Similarity
│   ├── mbti_matrix.py                # 16x16 MBTI personality logic & matrix
│   ├── matching_engine.py            # Hybrid profile matching engine
│   ├── feedback_learner.py           # Adaptive ML feedback weight learner
│   └── evaluate.py                   # Performance analysis & evaluation script
├── tests/
│   └── test_matcher.py               # Pytest suite verifying NLP, MBTI, & feedback modules
├── app.py                            # Streamlit web dashboard application
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation
Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Datasets
Generate `data/users.csv` (100 user profiles) and `data/feedback.csv` (800+ interaction rows):
```bash
python src/dataset_generator.py
```

### 3. Run Automated Unit Tests
```bash
python -m pytest tests/test_matcher.py
```

### 4. Run Performance Evaluation
Analyze recommendation accuracy improvement before vs. after feedback learning:
```bash
python src/evaluate.py
```

### 5. Launch Interactive Web App Demo
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

---

## 📊 Performance Analysis Summary
- **Baseline Static Match Acceptance Rate**: `57.67%`
- **Adaptive ML Feedback Acceptance Rate**: `66.33%`
- **Net Accuracy Improvement**: `+8.67%` (15.03% relative improvement)
- **Learned Global Weights**:
  - $w_1$ (NLP Text Similarity): `0.300`
  - $w_2$ (MBTI Compatibility): `0.493`
  - $w_3$ (Location Similarity): `0.207`
