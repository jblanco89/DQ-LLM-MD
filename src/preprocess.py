import json
import re
from typing import List, Dict
import numpy as np
# from config.config import EXPERTISE_RANKS

#To do: Add a config file to store constants and configurations
#To do: add a function to append the data to one file instead of creating a new one for each discussion
#To do: add a function to save the data to a csv/json file to be used in the next step of the pipeline (viualization or modeling) 

EXPERTISE_RANKS = {
    "Novice": 1,
    "Contributor": 2,
    "Expert": 3,
    "Master": 4,
    "Grandmaster": 5,
    None: 0 
}

def load_kaggle_data(path: str) -> List[Dict]:
    """Carga y normaliza datos de Kaggle"""
    with open(path, encoding='latin1') as f:
        data = json.load(f)
    
    cleaned_data = []
    for discussion_id, discussion in data.items():  # Iterate using .items()
        # Clean main discussion content
        discussion["content"] = clean_text(discussion["content"])
        
        # Clean comments if they exist
        if "comments" in discussion and discussion["comments"]:
            for comment_id, comment in discussion["comments"].items():
                comment["content"] = clean_text(comment["content"])
        
        cleaned_data.append(discussion)
    
    return cleaned_data

def clean_text(text: str) -> str:
    """Normaliza texto para análisis"""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'\s+', ' ', text)     # Espacios múltiples
    return text.strip()

def extract_structural_features(discussion: Dict) -> Dict:
    """Calcula métricas estructurales con manejo de casos faltantes"""
    # Handle missing comments
    comments = discussion.get("comments", {})
    
    # Calculate depth (0 if no comments)
    depth = calculate_thread_depth(comments) if comments else 0
    
    # Calculate expertise average (handle missing values)
    expertise_values = []
    for comment in comments.values():
        if "expertise" in comment:
            expertise_values.append(EXPERTISE_RANKS.get(comment["expertise"], 0))
    
    expertise_avg = np.mean(expertise_values) if expertise_values else 0
    
    # Calculate vote ratio (with protection against division by zero)
    votes = discussion.get("votes", 0)
    # Handle cases where n_comments is None, missing, or not a number
    try:
        n_comments = float(discussion.get("n_comments", 0))
    except (TypeError, ValueError):
        n_comments = 0
    
    # n_comments = discussion.get("n_comments", 0)
    vote_ratio = votes / (n_comments + 1e-6)  # Small epsilon to avoid division by zero
    
    features = {
    "depth": depth,
    "expertise_avg": float(expertise_avg),
    "vote_ratio": float(vote_ratio),
    "has_comments": bool(comments),  # Additional debug info
    "n_comments": len(comments)      # Additional debug info
    }
    return features
    
def calculate_thread_depth(comments: Dict) -> int:
    """Calcula la profundidad de la discusión en forma recursiva"""
    max_depth = 0
    for comment in comments.values():
        if "replies" in comment and comment["replies"]:
            depth = 1 + calculate_thread_depth(comment["replies"])
        else:
            depth = 1
        if depth > max_depth:
            max_depth = depth
    return max_depth   
    



if __name__ == "__main__":
    data = load_kaggle_data("./data/getting-started.json")
    # data = load_kaggle_data("./data/competition-hosting.json")
    for i, discussion in enumerate(data):
        features = extract_structural_features(discussion)
        print(f"Discussion ID: {i+1}, Features: {features}")
       
