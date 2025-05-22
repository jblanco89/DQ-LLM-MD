import re
import datetime
import logging
import json
import numpy as np
import pandas as pd
import ast


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
from openpyxl import Workbook

# to do: fix the ouput excel file information
# to do: calculate a mean of parameters stimated from the best model in every discussion
# to do: To use the best model parameters in lda analysis.py
# to do: process this tunning with mlflow pipeline
# to do: save the best model in the output folder



# Initialize tools
lemmatizer = WordNetLemmatizer()

def load_data_from_excel(path: str) -> List[Dict]:
    """Load and normalize data from Excel"""
    try:
        data = pd.read_excel(path)
        return data.to_dict(orient='records')
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return []

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
    # dictionary.filter_extremes(no_below=1, no_above=0.99)

    if len(dictionary) < 3:
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
    # min_topics = max(2, min(num_docs//4, 2))
    # max_topics = max(3, min(num_docs//2 + 1, 10))
    
    param_dist = {
        "num_topics": randint(2, 10),
        # "num_topics": randint(min_topics, max_topics),
        "alpha": ['symmetric', 'auto'],
        "eta": ['auto'],
        "passes": randint(5, 15),
        "chunksize": randint(100, 1000),
        "decay": [0.7, 0.5],
        "offset": [10, 12, 15],
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

def save_results_to_excel(results: Dict, path: str):
    """Save results to Excel"""
    try:
        with pd.ExcelWriter(path, engine='openpyxl', mode='a') as writer:
            for i, result in results.items():
                df = pd.concat([
                    pd.DataFrame({'Topics': [result['topics']]}),
                    pd.DataFrame({
                        'Coherence Score': [result['coherence_score']],
                        'Parameters': [result['parameters']],
                        'Topic Distribution': [result['topic_distribution']]
                    })
                ], axis=1)
                # df = df.explode('Topics').reset_index(drop=True)
                df.to_excel(writer, sheet_name=f"Discussion_{i}", index=False)
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise

def print_results(results: Dict):
    """Pretty print the results"""
    print(f"\n{'='*50}")
    print(f"Best Coherence Score: {results['coherence_score']:.4f}")
    print(f"Parameters Used: {results['parameters']}")
    
    print("\nTopic Probabilities and Terms:")
    for topic in results['topics']:
        print(f"\nTopic {topic['topic_id']} (Probability: {topic['probability']:.2%})")
        print("Top Terms:", ", ".join(topic['terms']))
    
    print("\nTotal Document Topic Distribution:")
    for topic_id, prob in results['topic_distribution']:
        print(f"Topic {topic_id}: {prob:.2%}")

def analyze_discussions(data: List[Dict], max_discussions: int = 5):
    results = {}
    for i, discussion in enumerate(data[:max_discussions]):
        title = discussion.get('title', f"Discussion {i}")
        print(f"\nAnalyzing: {title}")
        
        tuning_results = lda_hyperparameter_tuning(discussion, n_iter=3)
        if tuning_results:
            print_results(tuning_results)
            results[i] = tuning_results  # Using index as ID
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
    data_path = "src/results/processed_data.xlsx"  # Update with your path
    output_path = "src/results/lda_tuning_results.xlsx"
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True)
    
    output_file = Path(output_path)
    if not output_file.exists():
        # Create an empty Excel file with a default sheet.
        wb = Workbook()
        wb.save(output_file)
    try:
        discussions_data = load_data_from_excel(data_path)
        print(f"Loaded {len(discussions_data)} discussions")
        results = analyze_discussions(discussions_data, max_discussions=1)
        save_results_to_excel(results, output_path)
        logging.info(f"Results saved to {output_path}")    
    except Exception as e:
        logging.critical(f"Pipeline failed: {str(e)}")
        raise