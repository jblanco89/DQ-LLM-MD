import re
import gensim
import datetime
import logging
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import Phrases
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform
from pathlib import Path


# if python .\tunning\lda_tunning.py does not work, run this command in the terminal:
# python -m tunning.lda_tunning
from src.preprocess import load_kaggle_data

Path("tunning/logs").mkdir(parents=True, exist_ok=True)

now = datetime.datetime.now()
log_filename = f"fine_tune_lda_{now.strftime('%Y%m%d')}.log"

# Configure logging
logging.basicConfig(filename=f"tunning/logs/{log_filename}",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


# Initialize tools
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> list:
    """More conservative preprocessing that preserves key information"""
    if not text:
        return []
    
    # Keep more special characters that might be meaningful
    text = re.sub(r'!?\[\]?\(https?:\S+\)', ' SCREENSHOT ', text)  # Better image handling
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
    text = re.sub(r'```.*?```|`.*?`', ' CODE ', text, flags=re.DOTALL)
    text = re.sub(r'@\w+', lambda m: m.group().replace('@', 'USER_'), text)
    
    # Keep more punctuation that might be meaningful
    text = re.sub(r'[^a-zA-Z\s\-0-9_\.\?]', ' ', text)
    
    # Custom stopwords that preserves technical terms
    custom_stopwords = set(stopwords.words('english')) - {
        'how', 'what', 'why', 'when', 'which', 'where'
    }
    custom_stopwords.update({'hi', 'also', 'use'})  # Remove generic terms
    
    # Minimum length reduced to 2 for technical terms
    tokens = simple_preprocess(text, deacc=True, min_len=2, max_len=30)
    
    # Less aggressive filtering
    tokens = [lemmatizer.lemmatize(token) 
             for token in tokens 
             if token not in custom_stopwords]
    
    return tokens

def prepare_corpus(discussion: dict, min_docs: int = 1, min_tokens: int = 5) -> tuple:
    """Prepare corpus for LDA analysis from discussion data"""
    documents = []

    # Add main content with debug info
    if 'content' in discussion and discussion['content']:
        original_content = discussion['content']
        processed = preprocess_text(original_content)
        if processed:
            documents.append(processed)
            print(f"\nProcessing discussion: {discussion.get('title', 'No title')}")
            print(f"Original content sample: {original_content[:280]}...")
            print(f"Processed tokens sample: {processed[:len(processed)]}...")

    # Add comments
    comments = discussion.get("comments", {})
    for comment in comments.values():
        if 'content' in comment and comment['content']:
            processed = preprocess_text(comment['content'])
            if processed:
                documents.append(processed)

    # Skip if insufficient data
    if len(documents) < min_docs or sum(len(d) for d in documents) < min_tokens:
        print("Insufficient data after preprocessing")
        return None, None, None, None

    # Include technical terms and handle multi-word terms
    technical_terms = {
        'feature_engineering', 'auto_ml', 'data_centric', 
        'model_centric', 'random_search', 'grid_search', 'machine_learning',
        'deep_learning', 'transfer_learning', 'reinforcement_learning',
        'probabilistic_modeling', 'programming', 'data_science', 'data_analysis',
        'data_visualization', 'natural_language_processing', 'computer_vision',
        'time_series_analysis', 'classification', 'pytorch', 'word2vec',
        'fasttext', 'bert', 'gpt', 'transformer', 'doc2vec', 'cnn', 'rnn'
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

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(documents_phrased)
    
    # Adapt filter thresholds based on corpus size
    if len(documents) < 10:
        dictionary.filter_extremes(no_below=1, no_above=0.9)
    else:
        dictionary.filter_extremes(no_below=2, no_above=0.7)
    
    # Ensure we have enough terms
    if len(dictionary) < 5:
        dictionary.filter_extremes(no_below=1, no_above=0.95)
        if len(dictionary) < 5:
            print(f"Insufficient vocabulary: {len(dictionary)} terms")
            return None, None, None, None

    corpus = [dictionary.doc2bow(doc) for doc in documents_phrased]
    
    return documents_phrased, dictionary, corpus, len(documents)

def evaluate_lda_model(model, corpus, dictionary, texts):
    """Evaluate LDA model using coherence score"""
    try:
        # Calculate coherence using c_v measure
        coherence_model = CoherenceModel(
            model=model, 
            texts=texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        return coherence_score
    except Exception as e:
        print(f"Error calculating coherence: {str(e)}")
        return -1

def lda_hyperparameter_tuning(discussion: dict, n_iter: int = 10) -> list:
    """Perform hyperparameter tuning for LDA on a discussion"""
    # Prepare the corpus
    texts, dictionary, corpus, num_docs = prepare_corpus(discussion)
    if not corpus:
        return None
    
    print(f"Corpus prepared: {num_docs} documents, {len(dictionary)} unique terms")
    
    # Parameter search space
    param_dist = {
        # Adapt number of topics based on corpus size
        "num_topics": randint(max(1, min(num_docs // 3, 2)), 
                             min(max(num_docs // 2, 3), 10) + 1),
        "alpha": ['symmetric', 'asymmetric', 'auto'],
        "eta": ['symmetric', 'auto'],
        "passes": randint(5, 31),  # Number of passes through corpus
        "chunksize": randint(50, 2001),  # Batch size
        "decay": uniform(0.5, 0.49),  # Learning rate decay [0.5, 0.99]
        "offset": randint(1, 51),  # Learning offset
        "per_word_topics": [True],
        "random_state": [42]
    }
    
    # Sample parameters
    param_sets = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))
    
    # Track best model
    best_model = None
    best_score = -float('inf')
    best_params = None
    best_topics = None
    
    print(f"Running hyperparameter search with {n_iter} iterations...")
    
    # Train and evaluate models
    for i, params in enumerate(tqdm(param_sets)):
        try:
            # Train LDA model
            lda_model = gensim.models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                **params
            )
            
            # Evaluate model
            coherence_score = evaluate_lda_model(lda_model, corpus, dictionary, texts)
            
            # Track best model by coherence score
            print(f"Trial {i+1}/{n_iter}: Topics={params['num_topics']}, "
                  f"Alpha={params['alpha']}, Passes={params['passes']}, "
                  f"Coherence={coherence_score:.4f}")
            
            if coherence_score > best_score:
                best_score = coherence_score
                best_model = lda_model
                best_params = params
                # Get topics
                best_topics = [(i, " + ".join([t[0] for t in lda_model.show_topic(i, 5)])) 
                              for i in range(params["num_topics"])]
                
        except Exception as e:
            print(f"Error in trial {i+1}: {str(e)}")
            continue
    
    # Report results
    if best_model:
        print("\nHyperparameter Tuning Results:")
        print(f"Best Coherence Score: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")
        return best_topics
    else:
        print("No viable model found during tuning.")
        return None

if __name__ == '__main__':
    # Load data
    data_path = "./data/competition-hosting.json"  # Change as needed
    data = load_kaggle_data(data_path)
    
    # Analyze a subset of discussions
    for i, discussion in enumerate(data[:2]):
        print(f"\n{'='*80}\nAnalyzing discussion {i+1}: {discussion.get('title', 'No title')}")
        
        topics = lda_hyperparameter_tuning(discussion, n_iter=4)
        
        if topics:
            print("\nDetected topics:")
            for topic_id, topic_terms in topics:
                print(f"Topic {topic_id+1}: {topic_terms}")
        else:
            print("No valid topics extracted for this discussion")