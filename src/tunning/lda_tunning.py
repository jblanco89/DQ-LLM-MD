import re
import datetime
import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import Phrases, LdaModel
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint

# Initialize tools
lemmatizer = WordNetLemmatizer()

def load_json_data(path: str) -> Dict[str, Dict]:
    """Load JSON data with discussion ID as keys"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_text(text: str) -> List[str]:
    """Preprocess text with technical term preservation"""
    if not text:
        return []
    
    # Enhanced URL and special pattern handling
    text = re.sub(r'!?\[\]?\(https?:\S+\)', ' SCREENSHOT ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
    text = re.sub(r'```.*?```|`.*?`', ' CODE ', text, flags=re.DOTALL)
    text = re.sub(r'@\w+', lambda m: m.group().replace('@', 'USER_'), text)
    
    # Keep meaningful punctuation
    text = re.sub(r'[^a-zA-Z\s\-0-9_\.\?]', ' ', text)
    
    # Technical-aware stopwords
    custom_stopwords = set(stopwords.words('english')) - {
        'how', 'what', 'why', 'when', 'which', 'where', 'many'
    }
    custom_stopwords.update({'hi', 'also', 'use', 'like'})
    
    tokens = simple_preprocess(text, deacc=True, min_len=2, max_len=30)
    tokens = [lemmatizer.lemmatize(token) 
             for token in tokens 
             if token not in custom_stopwords]
    
    return tokens

def prepare_corpus(discussion: Dict) -> Tuple[Optional[List[List[str]]], 
                                            Optional[corpora.Dictionary], 
                                            Optional[List], 
                                            Optional[int]]:
    """Prepare corpus from discussion JSON structure"""
    documents = []
    
    # Process main content
    if discussion.get('content'):
        processed = preprocess_text(discussion['content'])
        if processed:
            documents.append(processed)
    
    # Process comments
    comments = discussion.get('comments', {})
    for comment_id, comment_data in comments.items():
        if comment_data.get('content'):
            processed = preprocess_text(comment_data['content'])
            if processed:
                documents.append(processed)
    
    # Skip if insufficient data
    if len(documents) < 1 or sum(len(d) for d in documents) < 5:
        return None, None, None, None
    
    # Technical terms preservation
    technical_terms = {
        'feature_engineering', 'dataset', 'machine_learning', 
        'data_analysis', 'model_training', 'data_visualization'
    }
    
    documents = [
        [term if term in technical_terms else word 
         for word in doc 
         for term in (word.replace(' ', '_'),)]
        for doc in documents
    ]
    
    # Phrase detection
    bigram = Phrases(documents, min_count=2, threshold=5)
    trigram = Phrases(bigram[documents], min_count=2, threshold=5)
    documents_phrased = [trigram[bigram[doc]] for doc in documents]
    
    # Create dictionary
    dictionary = corpora.Dictionary(documents_phrased)
    dictionary.filter_extremes(no_below=1, no_above=0.9)
    
    if len(dictionary) < 5:
        return None, None, None, None
    
    corpus = [dictionary.doc2bow(doc) for doc in documents_phrased]
    return documents_phrased, dictionary, corpus, len(documents)

def evaluate_lda_model(model: LdaModel, corpus: List, 
                      dictionary: corpora.Dictionary, 
                      texts: List[List[str]]) -> float:
    """Evaluate LDA model using coherence score"""
    try:
        coherence_model = CoherenceModel(
            model=model, 
            texts=texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    except Exception as e:
        logging.error(f"Coherence calculation failed: {str(e)}")
        return -1

def lda_hyperparameter_tuning(discussion: Dict, n_iter: int = 10) -> Optional[Dict]:
    """Returns dictionary with coherence score and topic probabilities"""
    texts, dictionary, corpus, num_docs = prepare_corpus(discussion)
    if not corpus:
        logging.warning("Skipped discussion - insufficient data")
        return None

    # Adaptive parameter ranges
    min_topics = max(2, min(num_docs//4, 2))
    max_topics = max(3, min(num_docs//2 + 1, 10))
    
    param_dist = {
        "num_topics": randint(min_topics, max_topics),
        "alpha": ['symmetric', 'auto'],
        "eta": ['auto'],
        "passes": randint(5, 15),
        "chunksize": randint(100, 1000),
        "decay": [0.7],
        "offset": [10, 12],
        "per_word_topics": [True, False],
        "random_state": [42]
    }

    best_model = None
    best_score = -float('inf')
    best_params = None

    for params in tqdm(list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))):
        try:
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                **params
            )
            
            score = evaluate_lda_model(model, corpus, dictionary, texts)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = params
                
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            continue

    if not best_model:
        return None

    # Get topic probabilities and terms
    topic_info = []
    for i in range(best_params["num_topics"]):
        topic_terms = best_model.show_topic(i, 10)
        topic_prob = sum(prob for _, prob in topic_terms)
        topic_info.append({
            "topic_id": i,
            "probability": topic_prob,
            "terms": [term for term, _ in topic_terms]
        })

    return {
        "coherence_score": best_score,
        "parameters": best_params,
        "topics": topic_info,
        "topic_distribution": best_model.get_document_topics(corpus[0])
    }

def print_results(results: Dict):
    """Pretty print the results"""
    print(f"\n{'='*50}")
    print(f"Best Coherence Score: {results['coherence_score']:.4f}")
    print(f"Parameters Used: {results['parameters']}")
    
    print("\nTopic Probabilities and Terms:")
    for topic in results['topics']:
        print(f"\nTopic {topic['topic_id']} (Probability: {topic['probability']:.2%})")
        print("Top Terms:", ", ".join(topic['terms']))
    
    print("\nExample Document Topic Distribution:")
    for topic_id, prob in results['topic_distribution']:
        print(f"Topic {topic_id}: {prob:.2%}")

# In your analyze_discussions function:
def analyze_discussions(data: Dict[str, Dict], max_discussions: int = 5):
    results = {}
    for discussion_id, discussion in list(data.items())[:max_discussions]:
        title = discussion.get('title', f"Discussion {discussion_id}")
        print(f"\nAnalyzing: {title}")
        
        tuning_results = lda_hyperparameter_tuning(discussion, n_iter=5)
        if tuning_results:
            print_results(tuning_results)
            results[discussion_id] = tuning_results
        else:
            print("No topics extracted")
    
    return results

if __name__ == '__main__':
    # Configure logging
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename=f"src/tunning/logs/lda_tuning_{datetime.datetime.now().strftime('%Y%m%d')}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load and process data
    data_path = "data/accomplishments.json"  # Update with your path
    try:
        discussions_data = load_json_data(data_path)
        print(f"Loaded {len(discussions_data)} discussions")
        
        results = analyze_discussions(discussions_data, max_discussions=2)
        
        # Simple fix - convert numpy floats to regular floats before saving
        def convert_floats(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(v) for v in obj]
            return obj


        # # Save results
        # with open("src/results/lda_topics.json", "w") as f:
        #     json.dump(convert_floats(results), f, indent=2)
            
    except Exception as e:
        logging.critical(f"Pipeline failed: {str(e)}")
        raise