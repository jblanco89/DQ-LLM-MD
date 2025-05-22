import re
import os
import json
import emoji
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import defaultdict

# to do: add more technical terms to the list

class TextCleaner:
    """Utility class for cleaning and normalizing text"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize text for analysis"""
        text = emoji.demojize(text, delimiters=(" :", ": "))
        text = text.replace(":", "")  # Ej: ":smiling_face:" → "smiling_face"
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
        text = re.sub(r'\s+', ' ', text)     # Multiple spaces
        
        # Keep more special characters that might be meaningful
        text = re.sub(r'!?\[\]?\(https?:\S+\)', ' SCREENSHOT ', text)
        text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
        text = re.sub(r'```.*?```|`.*?`', ' CODE ', text, flags=re.DOTALL)
        text = re.sub(r'@\w+', lambda m: m.group().replace('@', 'USER_'), text)
        
        # Keep more punctuation that might be meaningful
        text = re.sub(r'[^a-zA-Z\s\-0-9_\.\?]', ' ', text)
        return text.strip()

class DomainClassifier:
    """Classifies discussions/comments into domains based on tags or content"""
    
    def __init__(self, domain_tags: Dict[str, List[str]]):
        self.domain_tags = domain_tags
        self._build_tag_to_domain_mapping()
    
    def _build_tag_to_domain_mapping(self) -> None:
        """Build a reverse mapping from tags to domains"""
        self.tag_to_domain = {}
        for domain, domain_tags in self.domain_tags.items():
            for tag in domain_tags:
                self.tag_to_domain[tag.lower()] = domain
    
    def classify(self, item: Dict, use_tags: bool = True) -> List[str]:
        """Classify an item (discussion/comment) into domains"""
        if use_tags and item.get("tags") is not None:  # Check if tags exists and is not None
            return self._classify_by_tags(item["tags"])
        return self._classify_by_content(item)
    
    def _classify_by_tags(self, tags: List[str]) -> List[str]:
        """Classify using tags"""
        domains = set()
        for tag in tags:
            domain = self.tag_to_domain.get(tag.lower())
            if domain:
                domains.add(domain)
        return list(domains) if domains else ["General"]
    
    def _classify_by_content(self, item: Dict) -> List[str]:
        """Classify using content keywords"""
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()
        tokens = set(word_tokenize(text))
        
        domain_scores = defaultdict(int)
        for domain, domain_tags in self.domain_tags.items():
            keywords = set(tag.lower() for tag in domain_tags)
            domain_scores[domain] = len(tokens & keywords)
        
        # Get domains with at least 1 matching keyword
        domains = {domain for domain, score in domain_scores.items() if score >= 1}
        return list(domains) if domains else ["General"]
    
class StructuralFeatureExtractor:
    """Extracts structural features from discussions and comments"""
    
    def __init__(self, expertise_ranks: Dict[str, int]):
        self.expertise_ranks = expertise_ranks
    
    def extract_discussion_features(self, discussion: Dict) -> Dict:
        """Extract structural features from a discussion"""
        comments = discussion.get("comments", {})
        
        features = {
            "depth": self._calculate_thread_depth(comments),
            "expertise_avg": self._calculate_expertise_avg(comments),
            "vote_ratio": self._calculate_vote_ratio(discussion),
            "has_comments": bool(comments),
            "n_comments": len(comments)
        }
        return features
    
    def _calculate_thread_depth(self, comments: Dict) -> int:
        """Calculate discussion depth recursively"""
        max_depth = 0
        for comment in comments.values():
            if "replies" in comment and comment["replies"]:
                depth = 1 + self._calculate_thread_depth(comment["replies"])
            else:
                depth = 1
            if depth > max_depth:
                max_depth = depth
        return max_depth
    
    def _calculate_expertise_avg(self, comments: Dict) -> float:
        """Calculate average expertise score"""
        expertise_values = []
        for comment in comments.values():
            if "expertise" in comment:
                expertise_values.append(self.expertise_ranks.get(comment["expertise"], 0))
        return float(np.mean(expertise_values)) if expertise_values else 0.0
    
    def _calculate_vote_ratio(self, discussion: Dict) -> float:
        """Calculate vote ratio with protection against division by zero"""
        votes = discussion.get("votes", 0)
        try:
            n_comments = float(discussion.get("n_comments", 0))
        except (TypeError, ValueError):
            n_comments = 0
        return float(votes / (n_comments + 1e-6))

class DataProcessor:
    """Processes raw data into structured DataFrames"""
    
    def __init__(self, domain_tags: Dict[str, List[str]], expertise_ranks: Dict[str, int]):
        self.domain_classifier = DomainClassifier(domain_tags)
        self.feature_extractor = StructuralFeatureExtractor(expertise_ranks)
        self.text_cleaner = TextCleaner()
    
    def load_kaggle_data(self, path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load and normalize Kaggle data into discussions, comments, and features"""
        with open(path, encoding='latin1') as f:
            data = json.load(f)
        
        discussions, comments, features = [], [], []
        
        for discussion_id, discussion in data.items():
            # Process discussion
            discussion.update({
                "id": discussion_id,
                "content": self.text_cleaner.clean_text(discussion["content"]),
                "forum": os.path.splitext(os.path.basename(path))[0],
                "n_comments": len(discussion.get("comments", {})),
                "domains": self.domain_classifier.classify(discussion)
            })
            
            # Process features
            features.append({
                **self.feature_extractor.extract_discussion_features(discussion),
                "discussion_id": discussion_id
            })
            
            # Process comments
            for comment_id, comment in discussion.pop("comments", {}).items():
                comments.append({
                    "discussion_id": discussion_id,
                    "id": comment_id,
                    "content": self.text_cleaner.clean_text(comment["content"]),
                    **{k: v for k, v in comment.items() if k != "content"}
                })
            
            discussions.append(discussion)
        
        return discussions, comments, features
    
    def process_files(self, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process all JSON files in directory and return DataFrames"""
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} does not exist")
        
        all_data = ([], [], [])
        
        for filename in os.listdir(data_dir):
            print(f"Processing {filename}")
            file_data = self.load_kaggle_data(os.path.join(data_dir, filename))
            for all_list, new_items in zip(all_data, file_data):
                all_list.extend(new_items)
        
        return tuple(pd.DataFrame(data) for data in all_data)
    
    @staticmethod
    def save_to_excel(dataframes: Tuple[pd.DataFrame, ...], output_path: str) -> None:
        """Save DataFrames to Excel sheets"""
        sheet_names = ["Discussions", "Comments", "Features"]
        with pd.ExcelWriter(output_path) as writer:
            for df, name in zip(dataframes, sheet_names):
                df.to_excel(writer, sheet_name=name, index=False)
        print(f"Excel file saved to {output_path}")