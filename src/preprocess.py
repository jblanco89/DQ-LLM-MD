from config.config import EXPERTISE_RANKS
from config.config import DOMAINS_TAGS
import json
import re
import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import defaultdict

#To do: Add a config file to store constants and configurations [done]
#To do: add a function to append the data to one file instead of creating a new one for each discussion [done]
#To do: add a function to save the data to a csv/json file to be used in the next step of the pipeline (viualization or modeling) [done]
#To do: fix the issue related to tags and domains. The domains are not being matched correctly. [done]


def load_kaggle_data(path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load and normalize Kaggle data into discussions, comments, and features"""
    with open(path, encoding='latin1') as f:
        data = json.load(f)
    
    discussions, comments, features = [], [], []
    
    for discussion_id, discussion in data.items():
        # Process discussion
        discussion.update({
            "id": discussion_id,
            "content": clean_text(discussion["content"]),
            "forum": os.path.splitext(os.path.basename(path))[0],
            "n_comments": len(discussion.get("comments", {})),
            "domains": set_discussion_domain(discussion)
        })
        
        # Process features
        features.append({
            **extract_structural_features(discussion),
            "discussion_id": discussion_id
        })
        
        # Process comments
        for comment_id, comment in discussion.pop("comments", {}).items():
            comments.append({
                "discussion_id": discussion_id,
                "id": comment_id,
                "content": clean_text(comment["content"]),
                **{k: v for k, v in comment.items() if k != "content"}
            })
        
        discussions.append(discussion)
    
    return discussions, comments, features

def clean_text(text: str) -> str:
    """Normaliza texto para análisis"""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'\s+', ' ', text)     # Espacios múltiples
    
    # Keep more special characters that might be meaningful
    text = re.sub(r'!?\[\]?\(https?:\S+\)', ' SCREENSHOT ', text)  # Better image handling
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
    text = re.sub(r'```.*?```|`.*?`', ' CODE ', text, flags=re.DOTALL)
    text = re.sub(r'@\w+', lambda m: m.group().replace('@', 'USER_'), text)
    
    # Keep more punctuation that might be meaningful
    text = re.sub(r'[^a-zA-Z\s\-0-9_\.\?]', ' ', text)
    return text.strip()

def process_files(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process all JSON files in directory and return DataFrames"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    all_data = ([], [], [])
    
    for filename in os.listdir(data_dir):
        print(f"Processing {filename}")
        file_data = load_kaggle_data(os.path.join(data_dir, filename))
        for all_list, new_items in zip(all_data, file_data):
            all_list.extend(new_items)
    
    return tuple(pd.DataFrame(data) for data in all_data)

def save_to_excel(dataframes: Tuple[pd.DataFrame, ...], output_path: str) -> None:
    """Save DataFrames to Excel sheets"""
    sheet_names = ["Discussions", "Comments", "Features"]
    with pd.ExcelWriter(output_path) as writer:
        for df, name in zip(dataframes, sheet_names):
            df.to_excel(writer, sheet_name=name, index=False)
    print(f"Excel file saved to {output_path}")




def set_discussion_domain(discussion: Dict) -> List[str]:
    """Set the domains of the discussion based on tags or content.
    
    First tries tag-based classification, falls back to keyword matching in text.
    """
    tags = discussion.get("tags", [])
    domains = set()
    
    if tags:
        # Tag-based classification
        tag_to_domain = {}
        for domain, domain_tags in DOMAINS_TAGS.items():
            for tag in domain_tags:
                tag_to_domain[tag.lower()] = domain
        
        for tag in tags:
            domain = tag_to_domain.get(tag.lower())
            if domain:
                domains.add(domain)
    else:
        # Text-based keyword matching
        text = f"{discussion.get('title', '')} {discussion.get('content', '')}".lower()
        tokens = set(word_tokenize(text))
        
        domain_scores = defaultdict(int)
        for domain, domain_tags in DOMAINS_TAGS.items():
            # Convert domain_tags to set for intersection
            keywords = set(tag.lower() for tag in domain_tags)
            domain_scores[domain] = len(tokens & keywords)
        
        # Get domains with at least 2 matching keywords
        domains = {domain for domain, score in domain_scores.items() if score >= 1}
    
    return list(domains) if domains else ["General"]
    
    # return list(domains) if domains else ["General"]


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
    data_dir = "./data/"
    output_path = "./src/results/processed_data.xlsx"
    
    dfs = process_files(data_dir)
    save_to_excel(dfs, output_path)
       
