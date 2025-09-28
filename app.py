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
CORS(app, origins=["*"])

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# Enhanced Education Level Mapping with proper hierarchy
EDUCATION_HIERARCHY = {
    "class-10": 1,
    "class-12-science": 2, "class-12-commerce": 2, "class-12-arts": 2,
    "diploma-engineering": 3, "diploma-computer": 3, "diploma-mechanical": 3, 
    "diploma-civil": 3, "diploma-electrical": 3, "polytechnic": 3,
    "bsc-cs": 4, "bsc-it": 4, "bsc": 4, "bcom": 4, "ba": 4, "bba": 4, 
    "bmm": 4, "bdes": 4, "bca": 4,
    "btech-cse": 5, "btech-it": 5, "btech-ece": 5, "btech-mechanical": 5, 
    "btech-civil": 5, "btech-electrical": 5, "be": 5,
    "mca": 6, "msc-cs": 6, "msc-it": 6, "msc": 6, "mba": 6, "mcom": 6, 
    "ma": 6, "mdes": 6, "ca": 6, "cs": 6, "cma": 6, "law": 6,
    "mtech-cse": 7, "mtech-it": 7, "mtech-ece": 7, "mtech-mechanical": 7, 
    "mtech-civil": 7, "me": 7, "medical": 7,
    "phd-engineering": 8, "phd-computer": 8, "phd-management": 8, 
    "phd-science": 8, "phd-other": 8
}

# Education field specializations
EDUCATION_FIELDS = {
    "btech-cse": ["computer science", "software", "programming", "ai", "ml", "java", "python"],
    "btech-it": ["information technology", "software", "web", "database", "networking"],
    "btech-ece": ["electronics", "communication", "embedded", "vlsi", "hardware"],
    "bca": ["computer applications", "programming", "software development", "web", "java"],
    "mca": ["advanced programming", "software engineering", "database", "system design"],
    "bsc-cs": ["computer science", "programming", "algorithms", "data structures"],
    "mtech-cse": ["advanced software", "research", "ai", "machine learning", "data science"],
    "mba": ["management", "business", "marketing", "finance", "strategy"],
    "bcom": ["accounting", "finance", "business", "commerce", "economics"],
    "diploma-computer": ["basic programming", "computer basics", "web development"]
}

# Enhanced Skills Mapping with synonyms and related technologies
SKILL_SYNONYMS = {
    "python": ["python", "py", "django", "flask", "pandas", "numpy", "scikit-learn"],
    "java": ["java", "spring", "hibernate", "jsp", "servlets", "spring-boot"],
    "javascript": ["javascript", "js", "node", "nodejs", "react", "angular", "vue", "typescript"],
    "web-development": ["html", "css", "frontend", "backend", "fullstack", "web", "responsive"],
    "react": ["react", "reactjs", "jsx", "redux", "next.js"],
    "machine-learning": ["ml", "ai", "deep learning", "tensorflow", "pytorch", "sklearn", "data science"],
    "data-science": ["data analysis", "statistics", "visualization", "pandas", "numpy", "matplotlib"],
    "android": ["android", "kotlin", "java", "mobile development", "android studio"],
    "sql": ["sql", "mysql", "postgresql", "database", "queries", "rdbms"],
    "spring": ["spring", "spring-boot", "spring-mvc", "java", "rest-api"],
    "php": ["php", "laravel", "codeigniter", "wordpress", "web development"],
    "html": ["html", "html5", "markup", "web", "frontend"],
    "css": ["css", "css3", "styling", "bootstrap", "sass", "responsive"],
    "design": ["figma", "adobe-xd", "ui", "ux", "photoshop", "sketch"],
    "marketing": ["digital-marketing", "seo", "sem", "social-media", "content-marketing"]
}

# Enhanced Interest to Job Category Mapping
INTEREST_CATEGORIES = {
    "software-development": ["developer", "programmer", "software", "coding", "engineer", "backend", "fullstack"],
    "web-development": ["web", "frontend", "backend", "fullstack", "ui", "ux", "html", "css", "javascript"],
    "mobile-development": ["android", "ios", "mobile", "app", "kotlin", "swift"],
    "artificial-intelligence": ["ai", "ml", "machine learning", "data science", "nlp", "deep learning"],
    "data-analytics": ["data", "analyst", "analytics", "bi", "visualization", "statistics"],
    "ui-ux-design": ["designer", "ui", "ux", "graphic", "creative", "figma", "adobe"],
    "digital-marketing": ["marketing", "digital marketing", "seo", "social media", "content"],
    "finance": ["finance", "fintech", "accounting", "banking", "investment"],
    "java-development": ["java", "spring", "backend", "enterprise", "rest-api"],
    "python-development": ["python", "django", "flask", "automation", "scripting"],
    "database": ["sql", "mysql", "database", "data", "queries"]
}

# Enhanced Location Mapping
CITY_SYNONYMS = {
    "bengaluru": ["bangalore", "bengaluru", "blr", "karnataka"],
    "delhi": ["delhi", "new delhi", "ncr", "dl", "new-delhi"],
    "mumbai": ["mumbai", "bombay", "mum", "maharashtra"],
    "chennai": ["chennai", "madras", "maa", "tamil nadu"],
    "hyderabad": ["hyderabad", "secunderabad", "hyd", "telangana"],
    "pune": ["pune", "pun", "maharashtra"],
    "kolkata": ["kolkata", "calcutta", "cal", "west bengal"],
    "jaipur": ["jaipur", "pink city", "rajasthan"],
    "noida": ["noida", "greater noida", "uttar pradesh"],
    "gurgaon": ["gurgaon", "gurugram", "haryana"],
    "dehradun": ["dehradun", "doon", "ddun", "uttarakhand"],
    "ahmedabad": ["ahmedabad", "amdavad", "gujarat"],
    "lucknow": ["lucknow", "uttar pradesh"],
    "bhubaneswar": ["bhubaneswar", "odisha"],
    "goa": ["goa", "panaji"]
}

NEARBY_CITIES = {
    "bengaluru": ["pune", "hyderabad", "chennai", "coimbatore", "mysuru"],
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

# Load dataset
def load_dataset():
    """Load the internship dataset"""
    try:
        # Try to load from the provided CSV structure
        data = {
            'id': range(1, 1001),
            'title': [],
            'description': [],
            'skills': [],
            'location': [],
            'duration': [],
            'organization': []
        }
        
        # Sample data based on your CSV structure
        job_templates = [
            {
                'title': 'Java Backend Intern',
                'description': 'Develop REST APIs and services using Java Spring Boot.',
                'skills': 'Java;Spring;REST API',
                'organizations': ['CodeCrafters', 'TechCorp', 'InnovateLabs']
            },
            {
                'title': 'Android Developer Intern',
                'description': 'Build Android apps with Java/Kotlin.',
                'skills': 'Java;Kotlin;APIs',
                'organizations': ['MobileWorks', 'AppDev Inc', 'TechSolutions']
            },
            {
                'title': 'BCA Project Intern',
                'description': 'Help with college project using PHP, MySQL.',
                'skills': 'PHP;MySQL;HTML;CSS',
                'organizations': ['College Labs', 'EduTech', 'ProjectHub']
            },
            {
                'title': 'Machine Learning Intern',
                'description': 'Work on predictive analytics using Python and scikit-learn.',
                'skills': 'Python;Machine Learning;Statistics',
                'organizations': ['AI Labs', 'DataTech', 'MLSolutions']
            },
            {
                'title': 'Frontend React Intern',
                'description': 'Create responsive UI with React and TailwindCSS.',
                'skills': 'HTML;CSS;JavaScript;React',
                'organizations': ['WebStudio', 'FrontendPro', 'UIWorks']
            },
            {
                'title': 'Data Analyst Intern',
                'description': 'Perform data cleaning and visualization using Python and Excel.',
                'skills': 'Python;SQL;Excel',
                'organizations': ['DataWorks', 'Analytics Hub', 'InsightLabs']
            },
            {
                'title': 'Web Development Intern',
                'description': 'Front-end using React and backend Node.js.',
                'skills': 'HTML;CSS;JavaScript;React;Node',
                'organizations': ['WebSolutions', 'FullStack Inc', 'DevHub']
            },
            {
                'title': 'UI/UX Design Intern',
                'description': 'Design wireframes and prototypes using Figma and Adobe XD.',
                'skills': 'Figma;Adobe XD;Design',
                'organizations': ['DesignStudios', 'CreativeWorks', 'UILabs']
            },
            {
                'title': 'Marketing Intern',
                'description': 'Manage social media and content creation.',
                'skills': 'Communication;Marketing;Social Media',
                'organizations': ['MarketCo', 'BrandWorks', 'SocialHub']
            },
            {
                'title': 'Data Science Intern',
                'description': 'Work on ML models and data analysis.',
                'skills': 'Python;SQL;Machine Learning',
                'organizations': ['DataScience Labs', 'ML Analytics', 'TechData']
            }
        ]
        
        cities = ["Bengaluru", "Delhi", "Mumbai", "Chennai", "Hyderabad", "Pune", "Kolkata", 
                  "Jaipur", "Noida", "Gurgaon", "Dehradun", "Ahmedabad", "Lucknow", "Bhubaneswar", "Goa"]
        durations = ["1 month", "2 months", "3 months", "4 months", "5 weeks", "6 months"]
        
        # Generate 1000 internships
        for i in range(1000):
            template = job_templates[i % len(job_templates)]
            data['title'].append(template['title'])
            data['description'].append(template['description'])
            data['skills'].append(template['skills'])
            data['location'].append(np.random.choice(cities))
            data['duration'].append(np.random.choice(durations))
            data['organization'].append(np.random.choice(template['organizations']))
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return pd.DataFrame()

# Preprocessing functions
def normalize_location(location):
    """Normalize location names using synonyms"""
    if not location:
        return ""
    
    location = str(location).lower().strip()
    for city, synonyms in CITY_SYNONYMS.items():
        if location in synonyms:
            return city
    return location

def expand_skills(skills_list):
    """Expand skills using synonyms"""
    expanded = set()
    for skill in skills_list:
        skill_lower = skill.lower().strip()
        expanded.add(skill_lower)
        
        # Add synonyms
        for key, synonyms in SKILL_SYNONYMS.items():
            if skill_lower in synonyms or skill_lower == key:
                expanded.update(synonyms)
    
    return list(expanded)

def preprocess_dataset(df):
    """Enhanced preprocessing of the dataset"""
    if df.empty:
        return df
        
    df = df.copy()
    
    # Create skills list
    df["skills_list"] = df["skills"].apply(lambda x: 
        [s.strip().lower() for s in re.split(r"[;,]\s*", str(x)) if s.strip()] if x else []
    )
    
    # Expand skills with synonyms
    df["expanded_skills"] = df["skills_list"].apply(expand_skills)
    
    # Normalize locations
    df["location_normalized"] = df["location"].apply(normalize_location)
    
    # Create searchable text for better matching
    df["searchable_text"] = df.apply(lambda row: 
        f"{row.get('title', '')} {row.get('description', '')} {row.get('skills', '')}".lower(), 
        axis=1
    )
    
    # Infer education requirements
    df["education_requirement"] = df.apply(infer_education_requirement, axis=1)
    
    # Add experience levels
    df["experience_level"] = df["title"].apply(infer_experience_level)
    
    return df

def infer_education_requirement(row):
    """Infer minimum education requirement from job title and description"""
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    
    if any(word in text for word in ["bca project", "college project"]):
        return 4  # BCA level
    elif "machine learning" in text or "data science" in text:
        return 5  # Bachelor's degree
    elif "senior" in text or "lead" in text:
        return 6  # Master's or experience
    else:
        return 4  # Default undergraduate

def infer_experience_level(title):
    """Infer experience level from job title"""
    title_lower = title.lower()
    if "senior" in title_lower or "lead" in title_lower:
        return "Advanced"
    elif "junior" in title_lower or "intern" in title_lower:
        return "Beginner"
    else:
        return "Intermediate"

# Scoring functions
def compute_skill_score(user_skills, job_expanded_skills, user_education=""):
    """Enhanced skill matching with education context"""
    if not user_skills or not job_expanded_skills:
        return 0
    
    # Expand user skills
    expanded_user_skills = set(expand_skills(user_skills))
    job_skills_set = set(job_expanded_skills)
    
    # Calculate match score
    matched_skills = expanded_user_skills & job_skills_set
    if not job_skills_set:
        return 0
    
    # Base score calculation
    base_score = len(matched_skills) / len(job_skills_set)
    
    # Education boost
    education_boost = 1.0
    if user_education in EDUCATION_FIELDS:
        education_keywords = EDUCATION_FIELDS[user_education]
        if any(keyword in " ".join(job_expanded_skills) for keyword in education_keywords):
            education_boost = 1.2
    
    return min(base_score * education_boost, 1.0)

def compute_interest_score(user_interests, job_title, job_description=""):
    """Enhanced interest matching with categories"""
    if not user_interests:
        return 0
    
    text = f"{job_title} {job_description}".lower()
    score = 0
    total_weight = len(user_interests)
    
    for interest in user_interests:
        interest_lower = interest.lower().replace("-", " ").replace("_", " ")
        
        # Direct match in title (higher weight)
        if interest_lower in job_title.lower():
            score += 1.0
            continue
            
        # Direct match in description
        if interest_lower in text:
            score += 0.8
            continue
        
        # Category-based matching
        category_match = False
        for category, keywords in INTEREST_CATEGORIES.items():
            if interest_lower in category or category.replace("-", " ") in interest_lower:
                category_matches = sum(1 for keyword in keywords if keyword in text)
                if category_matches > 0:
                    score += min(0.6, category_matches * 0.2)
                    category_match = True
                    break
        
        # Fallback: partial string matching
        if not category_match:
            if any(word in text for word in interest_lower.split()):
                score += 0.3
    
    return score / total_weight if total_weight > 0 else 0

def compute_location_score(user_location, job_location):
    """Enhanced location matching with proximity"""
    if not user_location or not job_location:
        return 0.5  # Neutral if location not specified
    
    user_loc = normalize_location(user_location)
    job_loc = normalize_location(job_location)
    
    # Exact match
    if user_loc == job_loc:
        return 1.0
    
    # Remote work
    if "remote" in job_loc.lower() or user_loc == "remote":
        return 0.9
    
    # Nearby cities
    if user_loc in NEARBY_CITIES:
        nearby = NEARBY_CITIES[user_loc]
        if job_loc in nearby:
            return 0.7
    
    # Same state approximation (basic)
    if len(user_loc) > 3 and len(job_loc) > 3:
        if user_loc[:3] == job_loc[:3]:  # Simple same region check
            return 0.4
    
    return 0.1  # Some minimal score for different locations

def compute_education_score(user_education, required_education):
    """Score based on education level compatibility"""
    if not user_education:
        return 0.5  # Neutral if not specified
    
    user_level = EDUCATION_HIERARCHY.get(user_education.lower(), 4)
    required_level = required_education or 4
    
    if user_level >= required_level:
        return 1.0  # Qualified
    elif user_level >= required_level - 1:
        return 0.8  # Close match
    elif user_level >= required_level - 2:
        return 0.6  # Possible with experience
    else:
        return 0.3  # Under-qualified but not impossible

# Load and preprocess dataset
df = load_dataset()
if not df.empty:
    df = preprocess_dataset(df)
    logger.info(f"Dataset loaded and preprocessed: {len(df)} internships")
else:
    logger.error("Failed to load dataset")

# API Endpoints
@app.route("/recommend", methods=["POST"])
def recommend():
    """Main recommendation endpoint with enhanced matching"""
    try:
        body = request.get_json(force=True)
        logger.info(f"Received request: {body}")
        
        # Extract and validate input
        education = body.get("education", "").strip()
        skills = body.get("skills", [])
        interests = body.get("interests", [])
        location_pref = body.get("location", "").strip()
        top_k = max(1, min(20, int(body.get("top_k", 5))))
        
        # Process skills
        if isinstance(skills, str):
            skills = [s.strip() for s in re.split(r"[;,]\s*", skills) if s.strip()]
        else:
            skills = [str(s).strip() for s in skills if str(s).strip()]
        
        # Process interests
        if isinstance(interests, str):
            interests = [interests.strip()] if interests.strip() else []
        else:
            interests = [str(i).strip() for i in interests if str(i).strip()]
        
        # Validate input
        if not any([skills, interests, location_pref, education]):
            return jsonify({"error": "Please provide at least one filter criterion"}), 400
        
        # Compute recommendations
        results = []
        for idx, row in df.iterrows():
            try:
                # Calculate individual scores
                skill_score = compute_skill_score(skills, row.get("expanded_skills", []), education)
                interest_score = compute_interest_score(interests, row.get("title", ""), row.get("description", ""))
                location_score = compute_location_score(location_pref, row.get("location", ""))
                education_score = compute_education_score(education, row.get("education_requirement", 4))
                
                # Weighted total score
                weights = {
                    "skills": 0.40,
                    "interests": 0.25,
                    "location": 0.20,
                    "education": 0.15
                }
                
                total_score = (
                    weights["skills"] * skill_score +
                    weights["interests"] * interest_score +
                    weights["location"] * location_score +
                    weights["education"] * education_score
                )
                
                # Only include results with meaningful scores
                if total_score > 0.15:
                    # Generate explanation
                    reasons = []
                    if skill_score > 0.3:
                        matched_skills = list(set([s.lower() for s in skills]) & 
                                            set(row.get("expanded_skills", [])))[:3]
                        if matched_skills:
                            reasons.append(f"Skills match: {', '.join(matched_skills[:3])}")
                    
                    if interest_score > 0.3:
                        reasons.append("Strong interest alignment")
                    
                    if location_score > 0.7:
                        reasons.append(f"Excellent location match: {row.get('location', 'N/A')}")
                    elif location_score > 0.4:
                        reasons.append(f"Good location compatibility")
                    
                    if education_score > 0.8:
                        reasons.append("Perfect education fit")
                    
                    reason = "; ".join(reasons) if reasons else "Good overall profile match"
                    
                    result = {
                        "id": int(row.get("id", idx + 1)),
                        "title": row.get("title", "Internship Opportunity"),
                        "short_title": (row.get("title", "")[:60] + "...") if len(row.get("title", "")) > 60 else row.get("title", ""),
                        "description": row.get("description", "No description available"),
                        "short_description": (row.get("description", "")[:150] + "...") if len(row.get("description", "")) > 150 else row.get("description", ""),
                        "skills": row.get("skills", ""),
                        "key_skills": matched_skills[:5] if 'matched_skills' in locals() else [],
                        "location": row.get("location", "Not specified"),
                        "duration": row.get("duration", "Not specified"),
                        "organization": row.get("organization", "Various"),
                        "experience_level": row.get("experience_level", "Beginner"),
                        "apply_link": f"https://internships.gov.in/apply/{int(row.get('id', idx + 1))}",
                        "score": round(total_score, 3),
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
        
        # Sort and return top results
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        
        logger.info(f"Returning {len(results)} recommendations")
        return jsonify({
            "recommendations": results,
            "total_found": len(results),
            "search_criteria": {
                "skills": skills,
                "interests": interests,
                "location": location_pref,
                "education": education
            }
        })
        
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {e}")
        return jsonify({"error": "Internal server error occurred"}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(df),
        "version": "1.0"
    })

@app.route("/stats", methods=["GET"])
def stats():
    """Statistics endpoint"""
    try:
        if df.empty:
            return jsonify({"error": "No data available"}), 404
            
        stats_data = {
            "total_internships": len(df),
            "unique_organizations": len(df["organization"].unique()),
            "locations": sorted(df["location"].unique().tolist()),
            "skills_distribution": df["skills"].value_counts().head(10).to_dict(),
            "location_distribution": df["location"].value_counts().to_dict(),
            "duration_distribution": df["duration"].value_counts().to_dict()
        }
        return jsonify(stats_data)
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}")
        return jsonify({"error": "Unable to fetch statistics"}), 500

@app.route("/skills", methods=["GET"])
def get_skills():
    """Get available skills for autocomplete"""
    try:
        all_skills = set()
        for skills_str in df["skills"].dropna():
            skills_list = [s.strip() for s in re.split(r"[;,]\s*", skills_str) if s.strip()]
            all_skills.update(skills_list)
        
        return jsonify({
            "skills": sorted(list(all_skills)),
            "categories": list(SKILL_SYNONYMS.keys())
        })
    except Exception as e:
        logger.error(f"Error in skills endpoint: {e}")
        return jsonify({"error": "Unable to fetch skills"}), 500

@app.route("/locations", methods=["GET"])
def get_locations():
    """Get available locations"""
    try:
        return jsonify({
            "locations": sorted(df["location"].unique().tolist()),
            "major_cities": list(CITY_SYNONYMS.keys())
        })
    except Exception as e:
        logger.error(f"Error in locations endpoint: {e}")
        return jsonify({"error": "Unable to fetch locations"}), 500

# Run the application
if __name__ == "__main__":
    logger.info(f"Starting Enhanced AI Internship Recommendation System")
    logger.info(f"Dataset contains {len(df)} internships")
    app.run(host="0.0.0.0", port=5000, debug=True)