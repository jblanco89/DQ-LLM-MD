from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import Phrases
from gensim.utils import simple_preprocess
import pandas as pd
import ast
import numpy as np
from scipy.special import rel_entr

from typing import Dict, List, Optional, Tuple

# Initialize tools
lemmatizer = WordNetLemmatizer()

def load_processed_data(path: str) -> List[Dict]:
    """Load and normalize data from Excel"""
    try:
        data = pd.read_excel(path)
        return data.to_dict(orient='records')
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return []


def get_tokens(text: str) -> List[str]:
    """Preprocess text preserving technical terms"""
    if not text:
        return []

    # Reemplazo manual de términos técnicos antes de tokenizar
    technical_phrases = [
        "machine learning", "deep learning", "natural language processing", 
        "data science", "feature engineering", "computer vision"
    ]
    for phrase in technical_phrases:
        text = text.lower().replace(phrase, phrase.replace(" ", "_"))

    tokens = simple_preprocess(text, deacc=True, min_len=2, max_len=30)
    
    # Stopword filtering
    stop_words = set(stopwords.words("english")) - {"what", "how", "when", "why", "which", "where"}
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return tokens


def format_topic_terms(terms: List[Tuple[str, float]]) -> str:
    """Format topic terms with probabilities"""
    return " | ".join([f"{term} ({prob:.2f})" for term, prob in terms])

def lda_analysis(discussion: Dict) -> Tuple[List[Tuple[int, str]], List[Tuple[int, float]]]:
     """Improved LDA analysis with type checking"""
     documents = []
     
     content = discussion.get('content')
     if content:
        processed = get_tokens(content)
        if processed:
            documents.append(processed)
     elif content:
        print(f"Unexpected content type: {type(content)}")
 

    # Process comments with robust type handling
     comments = discussion.get("comments", {})
    
    # Handle case where comments might be stored as string
     if comments and isinstance(comments, str):
        try:
            comments = ast.literal_eval(comments) if comments.strip().startswith('{') else {}
        except (ValueError, SyntaxError):
            comments = {}
     else:
         comments = {}
     try:
        # Phrase detection and corpus preparation
        bigram = Phrases(documents, min_count=2, threshold=5)
        trigram = Phrases(bigram[documents], min_count=2, threshold=5)
        documents_phrased = [trigram[bigram[doc]] for doc in documents]
        
        dictionary = corpora.Dictionary(documents_phrased)
        # dictionary.filter_extremes(no_below=1, no_above=0.8)


        if len(dictionary) < 3:
            return None
            
        corpus = [dictionary.doc2bow(doc) for doc in documents_phrased]
        # print(corpus)
        
        # Model training
        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            chunksize=1000,
            decay=0.5,
            num_topics=4,
            random_state=42,
            passes=10,
            offset=1024,
            alpha='symmetric',
            eta='auto',
            iterations=100,
            per_word_topics=True,
        )
        
        # Extract and format topics
        topics = []
        for idx, topic in lda_model.print_topics(num_words=5, num_topics=-1):
            terms = [
                (term.split('*')[1].strip('"'), float(term.split('*')[0]))
                for term in topic.split('+')
            ]
            topics.append((idx, format_topic_terms(terms)))
        topic_distribution = lda_model.get_document_topics(corpus[0])
        return topics, topic_distribution
        
     except Exception as e:
        print(f"Error in LDA analysis: {str(e)[:200]}")
        return None

def print_discussion_topics(discussion: Dict, topics: List[Tuple[int, str]], topic_dist) -> None:
    """Pretty print discussion topics"""
    title = discussion.get('title', 'discussion')
    print(f"\n{'='*80}")
    print(f"Discussion: {title}")
    print(f"Author: {discussion.get('user', 'Unknown')}")
    print(f"Votes: {discussion.get('votes', 0)}")
    print(f"Comments: {discussion.get('n_comments', 0)}")
    print(f"Domains: {discussion.get('domains', {})}")
    print("\nDetected Topics:")
    for idx, topic in topics:
        print(f"  Topic {idx+1}: {topic}")
    print("\nTopic Distribution:")
    for idd, prob in topic_dist[:len(topic_dist)]:
        print(f"  Topic {idd+1}: {prob:.2%}")

if __name__ == '__main__':
    data = load_processed_data("src/results/processed_data.xlsx")
    if not data:
        print("No data loaded. Check file path and format.")
        exit()
    
    print(f"Loaded {len(data)} discussions")
    for discussion in data[0:10]:  # Process first 5 discussions
        topics, topic_dist = lda_analysis(discussion=discussion)  # Analyze first discussion
        print(topic_dist)
        if topics:
            print_discussion_topics(discussion=discussion, topics=topics, topic_dist=topic_dist)
        else:
            print(f"\nNo topics extracted for: {discussion.get('title', 'Untitled')}")
    

    # Este código es útil en procesamiento de lenguaje natural (NLP)
    # donde se quiere resumir la distribución de temas de un dominio 
    # así, podemos responder a la pregunta: 
    # "¿Qué proporción promedio de temas tiene cada dominio?"
    # y qué tan "atípico" es el documento (discusion) para su dominio
    # Dado que la lista de dominios parte de un enfoque heurístico del investigador,
    # surge la necesidad de restringir el peso del analisis KL al mínimo cuando se obtenga
    # el score final (DELIQ)

    def kl_divergence(p, q):
        # rel_entr computes the Kullback-Leibler divergence
        # between two probability distributions p and q
        # rel_entr stands for relative entropy
        p = np.array(p)
        q = np.array(q)
        return float(sum(rel_entr(p, q)))
    
    
    def dense_distribution(dist, num_topics):
        # Convert sparse distribution to dense format
        # dist is a list of tuples (index, probability)
        dense = [0.0] * num_topics
        for idx, prob in dist:
            dense[idx] = prob
        return dense
    
    def kl_content_vs_domain(data):
        content_records = []
        domain_topic_dists = {}
        domain_averages = {}

        # 1. Recolectamos las distribuciones para cada contenido y dominio
        for discussion in data[:len(data)]:  # Process all discussions
            result = lda_analysis(discussion)
            if result is None:
                continue
            topics, topic_dist = result
            dense = dense_distribution(topic_dist, len(topics))

            domains = discussion.get("domains", [])
            if isinstance(domains, str):
                try:
                    domains = ast.literal_eval(domains)
                except Exception:
                    continue

            for domain in domains:
                domain_topic_dists.setdefault(domain, []).append(dense)

            content_records.append({
                "title": discussion.get("title", "Untitled"),
                "domains": domains,
                "topic_dist": dense,
                "user": discussion.get("user", "Unknown"),
                "votes": discussion.get("votes", 0),
                "n_comments": discussion.get("n_comments", 0),
                "engagement": 0
            })

        # 2. Promedio de temas por dominio
        domain_averages = {
            domain: np.mean(dists, axis=0)
            for domain, dists in domain_topic_dists.items()
        }

        # 3. Calcular KL por contenido respecto a su(s) dominio(s)
        kl_results = []

        for record in content_records:
            for domain in record["domains"]:
                domain_avg = domain_averages.get(domain)
                if domain_avg is None:
                    continue
                divergence = kl_divergence(record["topic_dist"], domain_avg)
                positive_votes = max(record["votes"], 0)
                kl_results.append({
                    "title": record["title"],
                    "domain": domain,
                    "kl_divergence": divergence,
                    "user": record["user"],
                    "votes": positive_votes,
                    "n_comments": record["n_comments"],
                    "engagement": positive_votes + record["n_comments"] * 2,
                })

        df = pd.DataFrame(kl_results)
        print(df)
        df.to_excel("src/results/kl_content_vs_domain.xlsx", index=False)
        print("KL content vs domain exported to 'kl_content_vs_domain.xlsx'")

    kl_content_vs_domain(data)