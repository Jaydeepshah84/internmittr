# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import os
# import numpy as np
# import re

# app = Flask(__name__)
# CORS(app)

# # --- Load CSV dataset ---
# DATA_PATH = os.path.join("data", "internships.csv")
# if not os.path.exists(DATA_PATH):
#     alt = os.path.join(os.path.dirname(__file__), "internships.csv")
#     if os.path.exists(alt):
#         DATA_PATH = alt

# df = pd.read_csv(DATA_PATH).fillna("")

# # Ensure skills and locations are lowercase for matching
# df["skills_list"] = df["skills"].apply(lambda x: [s.strip().lower() for s in re.split(r"[;,]\s*", x)])
# df["location"] = df["location"].astype(str)

# # --- Nearby cities mapping (can expand as needed) ---
# NEARBY_CITIES = {
#     "bengaluru": ["pune", "hyderabad", "chennai", "coimbatore", "mysuru"],
#     "dehradun": ["haridwar", "rishikesh", "chandigarh", "lucknow"],
#     "delhi": ["noida", "gurgaon", "faridabad", "ghaziabad"],
#     "mumbai": ["pune", "navi mumbai", "thane", "surat"],
#     "chennai": ["bengaluru", "hyderabad", "coimbatore"],
#     "kolkata": ["bhubaneswar", "patna", "siliguri"],
#     "hyderabad": ["bengaluru", "chennai", "vijayawada"],
#     "pune": ["mumbai", "bengaluru", "goa"],
#     "jaipur": ["ajmer", "udaipur", "delhi"],
#     "noida": ["delhi", "gurgaon", "faridabad"],
# }

# # --- Helper functions ---
# def compute_skill_score(user_skills, job_skills):
#     if not user_skills or not job_skills:
#         return 0
#     matched = set(user_skills) & set(job_skills)
#     return len(matched) / max(len(job_skills), 1)

# def compute_interest_score(user_interests, job_title):
#     # Simple: if job title contains any interest keyword
#     if not user_interests:
#         return 0
#     score = 0
#     job_title_lower = job_title.lower()
#     for interest in user_interests:
#         if interest.lower() in job_title_lower:
#             score += 1
#     return score / max(len(user_interests), 1)

# def compute_location_score(user_location, job_location):
#     user_location_lower = user_location.lower()
#     job_location_lower = job_location.lower()
#     if user_location_lower == job_location_lower:
#         return 1.0
#     nearby = NEARBY_CITIES.get(user_location_lower, [])
#     if job_location_lower in nearby:
#         return 0.7
#     elif "remote" in job_location_lower:
#         return 0.5
#     return 0

# # --- Recommendation endpoint ---
# @app.route("/recommend", methods=["POST"])
# def recommend():
#     body = request.get_json(force=True)
    
#     # --- Extract profile ---
#     education = body.get("education", "")
#     skills = body.get("skills", [])
#     if isinstance(skills, str):
#         skills = [s.strip().lower() for s in re.split(r"[;,]\s*", skills)]
#     else:
#         skills = [s.lower() for s in skills]
        
#     interests = body.get("interests", [])
#     if isinstance(interests, str):
#         interests = [interests.lower()]
#     else:
#         interests = [i.lower() for i in interests]
        
#     location_pref = body.get("location", "").strip()
#     if not location_pref:
#         return jsonify({"error": "Location is required"}), 400
    
#     top_k = max(3, min(5, int(body.get("top_k", 5))))
    
#     # --- Compute scores ---
#     results = []
#     for _, row in df.iterrows():
#         skill_score = compute_skill_score(skills, row["skills_list"])
#         interest_score = compute_interest_score(interests, row["title"])
#         location_score = compute_location_score(location_pref, row["location"])
        
#         # Weighted total score
#         total_score = 0.5 * skill_score + 0.2 * interest_score + 0.3 * location_score
        
#         if total_score > 0:  # Only consider meaningful matches
#             results.append({
#                 "id": int(row.get("id", 0)),
#                 "title": row.get("title", ""),
#                 "short_title": row.get("title", "")[:60],
#                 "description": row.get("description", ""),
#                 "short_description": row.get("description", "")[:160],
#                 "skills": row.get("skills", ""),
#                 "key_skills": list(set(skills) & set(row["skills_list"])),
#                 "location": row.get("location", ""),
#                 "duration": row.get("duration", ""),
#                 "organization": row.get("organization", ""),
#                 "apply_link": f"https://example.org/apply/{int(row.get('id', 0))}",
#                 "score": round(total_score, 4),
#                 "reason": f"Skills matched: {', '.join(list(set(skills) & set(row['skills_list'])))}; Location relevance: {row.get('location')}"
#             })
    
#     # Sort by total_score descending and pick top_k
#     results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    
#     return jsonify(results)

# # --- Run Flask app ---
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)




from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import numpy as np
import re
from datetime import datetime
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:5500", "*"])

# --- Enhanced Education Level Mapping ---
EDUCATION_HIERARCHY = {
    "class-10": 1,
    "class-12-science": 2, "class-12-commerce": 2, "class-12-arts": 2,
    "diploma-engineering": 3, "diploma-computer": 3, "diploma-mechanical": 3, 
    "diploma-civil": 3, "diploma-electrical": 3, "polytechnic": 3,
    "btech-cse": 5, "btech-it": 5, "btech-ece": 5, "btech-mechanical": 5, 
    "btech-civil": 5, "btech-electrical": 5, "be": 5, "bca": 4, "bsc-cs": 4, 
    "bsc-it": 4, "bsc": 4, "bcom": 4, "ba": 4, "bba": 4, "bmm": 4, "bdes": 4,
    "mtech-cse": 7, "mtech-it": 7, "mtech-ece": 7, "mtech-mechanical": 7, 
    "mtech-civil": 7, "me": 7, "mca": 6, "msc-cs": 6, "msc-it": 6, "msc": 6, 
    "mba": 6, "mcom": 6, "ma": 6, "mdes": 6,
    "phd-engineering": 8, "phd-computer": 8, "phd-management": 8, "phd-science": 8, "phd-other": 8,
    "ca": 6, "cs": 6, "cma": 6, "law": 6, "medical": 7, "certification": 3
}

EDUCATION_FIELDS = {
    "btech-cse": ["computer science", "software", "programming", "ai", "ml"],
    "btech-it": ["information technology", "software", "web", "database"],
    "btech-ece": ["electronics", "communication", "embedded", "vlsi"],
    "mtech-cse": ["advanced software", "research", "ai", "machine learning"],
    "bca": ["computer applications", "programming", "software development"],
    "mca": ["advanced programming", "software engineering", "database"],
    "mba": ["management", "business", "marketing", "finance"],
    "bcom": ["accounting", "finance", "business", "commerce"],
    # Add more mappings as needed
}

# --- Enhanced Skills Mapping with Synonyms ---
SKILL_SYNONYMS = {
    "python": ["python", "py", "django", "flask", "pandas", "numpy"],
    "java": ["java", "spring", "hibernate", "jsp", "servlets"],
    "javascript": ["javascript", "js", "node", "nodejs", "react", "angular", "vue"],
    "web-development": ["html", "css", "frontend", "backend", "fullstack", "web"],
    "machine-learning": ["ml", "ai", "deep learning", "tensorflow", "pytorch", "sklearn"],
    "data-science": ["data analysis", "statistics", "visualization", "pandas", "numpy"],
    "mobile-development": ["android", "ios", "flutter", "react native", "kotlin", "swift"],
    "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "devops"],
    "cybersecurity": ["security", "penetration testing", "ethical hacking", "firewall"],
    "digital-marketing": ["seo", "sem", "social media", "content marketing", "analytics"]
}

# --- Enhanced Interest to Job Category Mapping ---
INTEREST_CATEGORIES = {
    "software-development": ["developer", "programmer", "software", "coding", "engineer"],
    "web-development": ["web", "frontend", "backend", "fullstack", "ui", "ux"],
    "mobile-development": ["android", "ios", "mobile", "app"],
    "artificial-intelligence": ["ai", "ml", "machine learning", "data science", "nlp"],
    "data-analytics": ["data", "analyst", "analytics", "bi", "visualization"],
    "design": ["designer", "ui", "ux", "graphic", "creative"],
    "marketing": ["marketing", "digital marketing", "seo", "social media"],
    "finance": ["finance", "fintech", "accounting", "banking"],
    "healthcare": ["healthcare", "medical", "pharma", "biotech"],
    "education": ["education", "training", "e-learning", "teaching"],
    "ecommerce": ["ecommerce", "retail", "marketplace", "shopping"],
    "gaming": ["game", "gaming", "unity", "unreal"],
    "research": ["research", "r&d", "innovation", "laboratory"],
    "consulting": ["consultant", "advisory", "strategy", "business"]
}

# --- Enhanced Location Mapping ---
CITY_SYNONYMS = {
    "bengaluru": ["bangalore", "bengaluru", "blr"],
    "delhi": ["delhi", "new delhi", "ncr", "dl"],
    "mumbai": ["mumbai", "bombay", "mum"],
    "chennai": ["chennai", "madras", "maa"],
    "hyderabad": ["hyderabad", "secunderabad", "hyd"],
    "pune": ["pune", "pun"],
    "kolkata": ["kolkata", "calcutta", "cal"],
    "jaipur": ["jaipur", "pink city"],
    "noida": ["noida", "greater noida"],
    "gurgaon": ["gurgaon", "gurugram"],
    "dehradun": ["dehradun", "doon", "ddun"]
}

NEARBY_CITIES = {
    "bengaluru": ["pune", "hyderabad", "chennai", "mysore", "coimbatore"],
    "delhi": ["noida", "gurgaon", "faridabad", "ghaziabad", "chandigarh"],
    "mumbai": ["pune", "thane", "navi mumbai", "surat", "nashik"],
    "chennai": ["bengaluru", "hyderabad", "coimbatore", "madurai"],
    "hyderabad": ["bengaluru", "chennai", "vijayawada", "visakhapatnam"],
    "pune": ["mumbai", "bengaluru", "nashik", "kolhapur"],
    "kolkata": ["bhubaneswar", "guwahati", "siliguri", "patna"],
    "jaipur": ["delhi", "udaipur", "jodhpur", "ajmer"],
    "noida": ["delhi", "gurgaon", "faridabad", "ghaziabad"],
    "gurgaon": ["delhi", "noida", "faridabad", "manesar"],
    "dehradun": ["haridwar", "rishikesh", "chandigarh", "lucknow"]
}

# --- Mock dataset creation (if CSV doesn't exist) ---
def create_mock_dataset():
    """Create a mock dataset if the CSV file doesn't exist"""
    companies = [
        "TCS", "Infosys", "Wipro", "HCL", "Cognizant", "Accenture", "IBM", "Microsoft",
        "Google", "Amazon", "Flipkart", "Paytm", "Zomato", "Swiggy", "BYJU'S", "Unacademy",
        "PhonePe", "Razorpay", "Freshworks", "Zoho", "InMobi", "Ola", "Uber", "BookMyShow"
    ]
    
    job_titles = [
        "Software Developer Intern", "Data Science Intern", "Web Developer Intern",
        "Mobile App Developer Intern", "UI/UX Designer Intern", "Digital Marketing Intern",
        "Business Analyst Intern", "Cybersecurity Intern", "Cloud Engineer Intern",
        "AI/ML Engineer Intern", "Frontend Developer Intern", "Backend Developer Intern",
        "Full Stack Developer Intern", "Data Analyst Intern", "Quality Assurance Intern",
        "DevOps Intern", "Product Manager Intern", "Content Writer Intern",
        "Graphic Designer Intern", "Finance Analyst Intern"
    ]
    
    cities = ["Bengaluru", "Delhi", "Mumbai", "Chennai", "Hyderabad", "Pune", "Kolkata", 
              "Jaipur", "Noida", "Gurgaon", "Remote", "Dehradun"]
    
    skills_pool = [
        "python,java,sql", "javascript,react,nodejs", "html,css,javascript",
        "java,spring,hibernate", "python,django,flask", "react,angular,vue",
        "aws,docker,kubernetes", "machine-learning,tensorflow,python",
        "data-science,pandas,numpy", "android,kotlin,java", "ios,swift,xcode",
        "ui-ux,figma,photoshop", "digital-marketing,seo,google-analytics",
        "cybersecurity,ethical-hacking,penetration-testing", "php,laravel,mysql"
    ]
    
    mock_data = []
    for i in range(100):
        company = np.random.choice(companies)
        title = np.random.choice(job_titles)
        city = np.random.choice(cities)
        skills = np.random.choice(skills_pool)
        
        mock_data.append({
            "id": i + 1,
            "title": f"{title} - {company}",
            "description": f"Exciting internship opportunity at {company} in {city}. "
                          f"Work on cutting-edge projects and gain hands-on experience. "
                          f"Perfect for students looking to advance their career in technology.",
            "skills": skills,
            "location": city,
            "duration": np.random.choice(["2 months", "3 months", "6 months"]),
            "organization": company,
            "category": np.random.choice(["Technology", "Finance", "Healthcare", "Education", "E-commerce"]),
            "experience_level": np.random.choice(["Beginner", "Intermediate", "Advanced"]),
            "stipend": np.random.choice(["10000", "15000", "20000", "25000", "Unpaid"]),
            "type": np.random.choice(["Full-time", "Part-time", "Remote", "Hybrid"])
        })
    
    return pd.DataFrame(mock_data)

# --- Load or create dataset ---
try:
    DATA_PATH = os.path.join("data", "internships.csv")
    if not os.path.exists(DATA_PATH):
        DATA_PATH = os.path.join(os.path.dirname(__file__), "internships.csv")
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH).fillna("")
        logger.info(f"Loaded dataset with {len(df)} internships from {DATA_PATH}")
    else:
        df = create_mock_dataset()
        logger.info(f"Created mock dataset with {len(df)} internships")
        
except Exception as e:
    logger.warning(f"Error loading dataset: {e}. Creating mock dataset.")
    df = create_mock_dataset()

# --- Preprocess dataset ---
def preprocess_dataset(df):
    """Enhanced preprocessing of the dataset"""
    df = df.copy()
    
    # Create skills list with enhanced matching
    df["skills_list"] = df["skills"].apply(lambda x: 
        [s.strip().lower() for s in re.split(r"[;,]\s*", str(x))] if x else []
    )
    
    # Normalize locations
    df["location_normalized"] = df["location"].apply(normalize_location)
    
    # Create searchable text for better matching
    df["searchable_text"] = df.apply(lambda row: 
        f"{row.get('title', '')} {row.get('description', '')} {row.get('skills', '')}".lower(), 
        axis=1
    )
    
    # Add derived fields
    df["education_requirement"] = df.apply(infer_education_requirement, axis=1)
    df["experience_level"] = df.get("experience_level", "Beginner")
    
    return df

def normalize_location(location):
    """Normalize location names using synonyms"""
    location = str(location).lower().strip()
    for city, synonyms in CITY_SYNONYMS.items():
        if location in synonyms:
            return city
    return location

def infer_education_requirement(row):
    """Infer minimum education requirement from job title and description"""
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    
    if any(word in text for word in ["senior", "lead", "architect", "principal"]):
        return 6  # Graduate level
    elif any(word in text for word in ["junior", "entry", "fresher", "intern"]):
        return 4  # Undergraduate
    elif any(word in text for word in ["experienced", "3+ years", "5+ years"]):
        return 5  # Bachelor's + experience
    else:
        return 4  # Default to undergraduate

# Preprocess the dataset
df = preprocess_dataset(df)

# --- Enhanced scoring functions ---
def compute_skill_score(user_skills, job_skills, user_education=""):
    """Enhanced skill matching with synonyms and education context"""
    if not user_skills or not job_skills:
        return 0
    
    # Expand skills with synonyms
    expanded_user_skills = set()
    for skill in user_skills:
        expanded_user_skills.add(skill.lower())
        if skill.lower() in SKILL_SYNONYMS:
            expanded_user_skills.update(SKILL_SYNONYMS[skill.lower()])
    
    expanded_job_skills = set()
    for skill in job_skills:
        expanded_job_skills.add(skill.lower())
        if skill.lower() in SKILL_SYNONYMS:
            expanded_job_skills.update(SKILL_SYNONYMS[skill.lower()])
    
    # Calculate match score
    matched_skills = expanded_user_skills & expanded_job_skills
    if not expanded_job_skills:
        return 0
    
    base_score = len(matched_skills) / len(expanded_job_skills)
    
    # Boost score based on education relevance
    education_boost = 1.0
    if user_education in EDUCATION_FIELDS:
        education_keywords = EDUCATION_FIELDS[user_education]
        if any(keyword in " ".join(job_skills) for keyword in education_keywords):
            education_boost = 1.2
    
    return min(base_score * education_boost, 1.0)

def compute_interest_score(user_interests, job_title, job_description=""):
    """Enhanced interest matching with categories"""
    if not user_interests:
        return 0
    
    text = f"{job_title} {job_description}".lower()
    score = 0
    total_weight = 0
    
    for interest in user_interests:
        weight = 1.0
        interest_lower = interest.lower()
        
        # Direct match
        if interest_lower in text:
            score += weight * 1.0
        
        # Category match
        if interest_lower in INTEREST_CATEGORIES:
            categories = INTEREST_CATEGORIES[interest_lower]
            category_matches = sum(1 for cat in categories if cat in text)
            if category_matches > 0:
                score += weight * (category_matches / len(categories))
        
        total_weight += weight
    
    return score / total_weight if total_weight > 0 else 0

def compute_location_score(user_location, job_location):
    """Enhanced location matching with synonyms and proximity"""
    if not user_location or not job_location:
        return 0
    
    user_loc = normalize_location(user_location)
    job_loc = normalize_location(job_location)
    
    # Exact match
    if user_loc == job_loc:
        return 1.0
    
    # Remote work
    if "remote" in job_loc or user_loc == "remote":
        return 0.8
    
    # Nearby cities
    if user_loc in NEARBY_CITIES:
        nearby = NEARBY_CITIES[user_loc]
        if job_loc in nearby:
            return 0.6
    
    # Same state (simplified - can be enhanced)
    user_parts = user_loc.split()
    job_parts = job_loc.split()
    if len(user_parts) > 1 and len(job_parts) > 1:
        if user_parts[-1] == job_parts[-1]:  # Same state
            return 0.4
    
    return 0

def compute_education_score(user_education, required_education):
    """Score based on education level compatibility"""
    if not user_education or not required_education:
        return 0.5  # Neutral if unknown
    
    user_level = EDUCATION_HIERARCHY.get(user_education, 0)
    required_level = required_education
    
    if user_level >= required_level:
        return 1.0  # Qualified
    elif user_level >= required_level - 1:
        return 0.7  # Close match
    else:
        return 0.3  # Under-qualified but possible

def compute_text_similarity(user_profile_text, job_text):
    """Compute text similarity using TF-IDF and cosine similarity"""
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        texts = [user_profile_text, job_text]
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0

# --- Main recommendation endpoint ---
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        body = request.get_json(force=True)
        logger.info(f"Received recommendation request: {body}")
        
        # Extract and validate input
        education = body.get("education", "").strip()
        skills = body.get("skills", [])
        interests = body.get("interests", [])
        location_pref = body.get("location", "").strip()
        top_k = max(1, min(20, int(body.get("top_k", 5))))
        
        # Validate required fields
        if not any([education, skills, interests, location_pref]):
            return jsonify({"error": "Please provide at least one filter criterion"}), 400
        
        # Process skills
        if isinstance(skills, str):
            skills = [s.strip().lower() for s in re.split(r"[;,]\s*", skills) if s.strip()]
        else:
            skills = [str(s).strip().lower() for s in skills if str(s).strip()]
        
        # Process interests
        if isinstance(interests, str):
            interests = [interests.strip().lower()]
        else:
            interests = [str(i).strip().lower() for i in interests if str(i).strip()]
        
        # Create user profile text for similarity matching
        user_profile_text = f"{education} {' '.join(skills)} {' '.join(interests)} {location_pref}"
        
        # Compute scores for all internships
        results = []
        for idx, row in df.iterrows():
            try:
                # Individual scores
                skill_score = compute_skill_score(skills, row["skills_list"], education)
                interest_score = compute_interest_score(interests, 
                                                      row.get("title", ""), 
                                                      row.get("description", ""))
                location_score = compute_location_score(location_pref, row.get("location", ""))
                education_score = compute_education_score(education, row.get("education_requirement", 4))
                text_similarity = compute_text_similarity(user_profile_text, row.get("searchable_text", ""))
                
                # Weighted total score
                weights = {
                    "skills": 0.35,
                    "interests": 0.25,
                    "location": 0.20,
                    "education": 0.10,
                    "text_similarity": 0.10
                }
                
                total_score = (
                    weights["skills"] * skill_score +
                    weights["interests"] * interest_score +
                    weights["location"] * location_score +
                    weights["education"] * education_score +
                    weights["text_similarity"] * text_similarity
                )
                
                # Only include results with meaningful scores
                if total_score > 0.1:  # Minimum threshold
                    # Generate reason
                    reasons = []
                    if skill_score > 0.3:
                        matched_skills = list(set(skills) & set(row["skills_list"]))[:3]
                        if matched_skills:
                            reasons.append(f"Skills match: {', '.join(matched_skills)}")
                    
                    if interest_score > 0.3:
                        reasons.append("Strong interest alignment")
                    
                    if location_score > 0.5:
                        reasons.append(f"Great location match: {row.get('location', 'N/A')}")
                    elif location_score > 0:
                        reasons.append(f"Location compatible: {row.get('location', 'N/A')}")
                    
                    if education_score > 0.8:
                        reasons.append("Perfect education fit")
                    
                    reason = "; ".join(reasons) if reasons else "Good overall match for your profile"
                    
                    # Create result object
                    result = {
                        "id": int(row.get("id", idx)),
                        "title": row.get("title", "Internship Opportunity"),
                        "short_title": row.get("title", "Internship Opportunity")[:80] + "..." if len(row.get("title", "")) > 80 else row.get("title", "Internship Opportunity"),
                        "description": row.get("description", ""),
                        "short_description": row.get("description", "")[:200] + "..." if len(row.get("description", "")) > 200 else row.get("description", ""),
                        "skills": row.get("skills", ""),
                        "key_skills": list(set(skills) & set(row["skills_list"]))[:5],  # Top 5 matched skills
                        "location": row.get("location", "Not specified"),
                        "duration": row.get("duration", "Not specified"),
                        "organization": row.get("organization", "Various"),
                        "category": row.get("category", "General"),
                        "experience_level": row.get("experience_level", "Beginner"),
                        "stipend": row.get("stipend", "Not specified"),
                        "type": row.get("type", "Full-time"),
                        "apply_link": f"https://internmitrr.gov.in/apply/{int(row.get('id', idx))}",
                        "score": round(total_score, 4),
                        "reason": reason,
                        "match_breakdown": {
                            "skills": round(skill_score, 2),
                            "interests": round(interest_score, 2),
                            "location": round(location_score, 2),
                            "education": round(education_score, 2),
                            "overall": round(total_score, 2)
                        }
                    }
                    
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        # Sort by score and return top results
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        
        logger.info(f"Returning {len(results)} recommendations")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {e}")
        return jsonify({"error": "Internal server error occurred"}), 500

# --- Health check endpoint ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(df),
        "version": "2.0"
    })

# --- Statistics endpoint ---
@app.route("/stats", methods=["GET"])
def stats():
    try:
        stats_data = {
            "total_internships": len(df),
            "unique_companies": len(df["organization"].unique()) if "organization" in df.columns else 0,
            "locations": list(df["location"].unique()) if "location" in df.columns else [],
            "top_skills": df["skills"].value_counts().head(10).to_dict() if "skills" in df.columns else {},
            "categories": df["category"].value_counts().to_dict() if "category" in df.columns else {}
        }
        return jsonify(stats_data)
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}")
        return jsonify({"error": "Unable to fetch statistics"}), 500

# --- Run Flask app ---
if __name__ == "__main__":
    logger.info(f"Starting InternMitrr API server with {len(df)} internships")
    app.run(host="0.0.0.0", port=5000, debug=True)