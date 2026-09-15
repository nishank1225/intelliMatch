import os
import random
import pandas as pd
from datetime import datetime, timedelta

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Predefined realistic datasets for generation
FIRST_NAMES = [
    "Rahul", "Priya", "Ananya", "Vikram", "Sneha", "Arjun", "Rhea", "Karan", "Pooja", "Aditya",
    "Siddharth", "Meera", "Rohan", "Tanvi", "Nikhil", "Divya", "Kabir", "Neha", "Aarav", "Ishita",
    "Alex", "Sarah", "David", "Emily", "Michael", "Jessica", "James", "Laura", "Daniel", "Samantha",
    "Chris", "Rachel", "Matthew", "Hannah", "Andrew", "Ashley", "Joshua", "Megan", "Ryan", "Lauren"
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Mehta", "Rao", "Nair", "Gupta", "Joshi", "Singhania", "Kapoor",
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"
]

CITIES = [
    "Bangalore, India", "Mumbai, India", "Delhi, India", "Hyderabad, India", "Pune, India",
    "San Francisco, USA", "New York, USA", "Seattle, USA", "London, UK", "Austin, USA"
]

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"
]

PROFESSIONS = [
    ("Data Analyst", "Analytical", ["SQL", "Python", "Tableau", "Excel", "Data Mining"]),
    ("Software Engineer", "Technical", ["Python", "Java", "React", "System Design", "Cloud"]),
    ("Product Manager", "Leadership", ["Roadmapping", "Agile", "User Research", "Strategy", "Analytics"]),
    ("UX Designer", "Creative", ["Figma", "User Testing", "Prototyping", "Design Systems", "Wireframing"]),
    ("Data Scientist", "Analytical", ["Machine Learning", "Deep Learning", "Python", "Statistics", "PyTorch"]),
    ("Marketing Specialist", "Creative", ["SEO", "Content Strategy", "Social Media", "Brand Management", "Analytics"]),
    ("Financial Analyst", "Analytical", ["Financial Modeling", "Excel", "Valuation", "Risk Analysis", "Forecasting"]),
    ("HR Specialist", "Leadership", ["Talent Acquisition", "Employee Engagement", "Performance Management", "Culture", "Onboarding"]),
    ("DevOps Engineer", "Technical", ["Docker", "Kubernetes", "CI/CD", "AWS", "Terraform"]),
    ("Business Analyst", "Analytical", ["Requirements Gathering", "SQL", "Process Optimization", "Agile", "BPMN"])
]

SUMMARY_TEMPLATES = [
    "{role} with {exp} years of experience in {domain}, passionate about leveraging {skill1} and {skill2} to solve complex problems and drive business growth.",
    "Results-driven {role} boasting {exp} years in the industry. Specialized in {skill1}, {skill2}, and scalable system design with a strong track record of success.",
    "Dedicated {role} with {exp} years of background in innovative environments. Focused on building high-impact solutions using {skill1} and modern {skill2} frameworks.",
    "Analytical and creative {role} with {exp} years of experience. Proven expertise in {skill1}, cross-functional leadership, and delivering value through {skill2}."
]

ABOUT_ME_TEMPLATES = [
    "I thrive in collaborative, fast-paced team environments where continuous learning is celebrated. Outside of work, I enjoy mentoring juniors, reading tech blogs, and staying active.",
    "Curious problem-solver who values transparent communication, empathy, and intellectual rigor. Passionate about community building, sustainability, and exploring new tech trends.",
    "Enthusiastic professional with a growth mindset. I love brainstorming innovative ideas over coffee, participating in hackathons, and hiking on weekend getaways.",
    "Structured thinker and creative innovator who enjoys tackling ambiguous challenges. I value work-life harmony, mindfulness, and engaging in deep philosophical discussions."
]

INTERESTS_POOL = [
    "Data", "Fitness", "Reading", "Teaching", "Hiking", "Travel", "Coding", "Music", "Photography",
    "Chess", "Cooking", "Gaming", "Writing", "Public Speaking", "Mentorship", "Startups", "AI Ethics"
]

def generate_users(n=100):
    users = []
    for i in range(1, n + 1):
        user_id = f"U{i:03d}"
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        age = random.randint(22, 45)
        location = random.choice(CITIES)
        role, archetype, skills = random.choice(PROFESSIONS)
        exp = random.randint(1, 15)
        
        if archetype == "Analytical":
            mbti = random.choice(["INTJ", "INTP", "ISTJ", "ENTP", "ENTJ"])
        elif archetype == "Leadership":
            mbti = random.choice(["ENTJ", "ENFJ", "ESTJ", "ESFJ"])
        elif archetype == "Creative":
            mbti = random.choice(["INFP", "ENFP", "ISFP", "INFJ"])
        else:
            mbti = random.choice(MBTI_TYPES)

        skill1, skill2 = random.sample(skills, 2)
        summary_template = random.choice(SUMMARY_TEMPLATES)
        prof_summary = summary_template.format(
            role=role, exp=exp, domain=role.split()[0].lower() + " solutions", skill1=skill1, skill2=skill2
        )
        
        about_me = random.choice(ABOUT_ME_TEMPLATES)
        interests = ", ".join(random.sample(INTERESTS_POOL, random.randint(3, 5)))
        
        users.append({
            "user_id": user_id,
            "name": name,
            "age": age,
            "location": location,
            "profession": role,
            "experience_years": exp,
            "professional_summary": prof_summary,
            "about_me": about_me,
            "mbti": mbti,
            "interests": interests
        })
    
    df = pd.DataFrame(users)
    output_path = os.path.join(DATA_DIR, "users.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} user profiles at {output_path}")
    return df

def generate_feedback(users_df, min_interactions_per_user=6):
    feedback = []
    user_ids = users_df["user_id"].tolist()
    start_date = datetime(2025, 1, 1)
    
    user_mbti_map = dict(zip(users_df["user_id"], users_df["mbti"]))
    user_prof_map = dict(zip(users_df["user_id"], users_df["profession"]))
    user_loc_map = dict(zip(users_df["user_id"], users_df["location"]))
    
    for u1 in user_ids:
        k = random.randint(min_interactions_per_user, 10)
        candidates = [u for u in user_ids if u != u1]
        sampled_candidates = random.sample(candidates, k)
        
        pref_type = random.choice(["profession", "mbti", "location", "balanced"])
        
        for idx, u2 in enumerate(sampled_candidates):
            action_prob = 0.5
            if pref_type == "profession" and user_prof_map[u1] == user_prof_map[u2]:
                action_prob += 0.35
            if pref_type == "mbti" and user_mbti_map[u1] == user_mbti_map[u2]:
                action_prob += 0.35
            if pref_type == "location" and user_loc_map[u1] == user_loc_map[u2]:
                action_prob += 0.30
                
            action = 1 if random.random() < min(0.9, action_prob) else 0
            ts = start_date + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
            
            feedback.append({
                "user_id": u1,
                "matched_user_id": u2,
                "action": action,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")
            })
            
    df_fb = pd.DataFrame(feedback)
    output_path = os.path.join(DATA_DIR, "feedback.csv")
    df_fb.to_csv(output_path, index=False)
    print(f"Generated {len(df_fb)} feedback interactions at {output_path}")
    return df_fb

if __name__ == "__main__":
    users_df = generate_users(100)
    generate_feedback(users_df, min_interactions_per_user=6)
