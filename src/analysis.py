
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import Phrases
from gensim.utils import simple_preprocess
from preprocess import load_kaggle_data

# to do: check if nltk data is available, if not download it
# to do: off-topic handling. Exclude them maybe by checking is_appreciation or is_deleted flags
# to do: manage hieriarchical comments as elegible for topic analysis optionally
# to do: consider to use an LDA alternative different from Bertopic but with similar performance or better. 
# to do: consider to merge here preprocessing text with preprocessing functions in preprocess.py


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

def lda_analysis(discussion: dict) -> list:
    """More robust LDA analysis with fallback options"""
    documents = []

    # Add main content with debug info
    if 'content' in discussion and discussion['content']:
        original_content = discussion['content']
        processed = preprocess_text(original_content)
        if processed:
            documents.append(processed)
            print(f"\nProcessing discussion: {discussion.get('title', 'No title')}")
            print(f"Original content sample: {original_content[:280]}...")
            print(f"Processed tokens: {processed}")

    # Add comments
    comments = discussion.get("comments", {})
    for comment in comments.values():
        if 'content' in comment and comment['content']:
            processed = preprocess_text(comment['content'])
            if processed:
                documents.append(processed)

    # Skip if insufficient data
    if len(documents) < 1 or sum(len(d) for d in documents) < 5:
        print("Insufficient data after preprocessing")
        return None
    
    technical_terms = {
    'feature_engineering', 'auto_ml', 'data_centric', 
    'model_centric', 'random_search', 'grid_search', 'machine_learning',
    'deep_learning', 'transfer_learning', 'reinforcement_learning','probabilistic_modeling',
    'programming', 'data_science', 'data_analysis', 'data_visualization',
    'natural_language_processing', 'computer_vision', 'time_series_analysis', 'classification',
    'pytorch', 'word-to-vecto','word2vec', 'fasttext', 'bert', 'gpt', 'transformer','doc2vec','cnn','rnn'
    }
    documents = [
    [term if term in technical_terms else word 
     for word in doc 
     for term in (word.replace(' ', '_'),)]
    for doc in documents
    ]

    try:
        # Phrase detection with lower thresholds
        bigram = Phrases(documents, min_count=2, threshold=5)
        trigram = Phrases(bigram[documents], min_count=2, threshold=5)
        documents_phrased = [trigram[bigram[doc]] for doc in documents]

        # More permissive dictionary
        dictionary = corpora.Dictionary(documents_phrased)
        # Relax filters when document count is low
        if len(documents) < 10:
            dictionary.filter_extremes(no_below=1, no_above=0.9)
        else:
            dictionary.filter_extremes(no_below=2, no_above=0.7)
        
        # Ensure we have enough words
        if len(dictionary) < 5:
            dictionary.filter_extremes(no_below=1, no_above=0.9) 
            print(f"Insufficient vocabulary: {len(dictionary)} terms")
            return None

        corpus = [dictionary.doc2bow(doc) for doc in documents_phrased]
        
        # Adaptive topic number
        num_topics = min(2, max(1, len(documents) // 2))  # At least 1, at most 2 topics
        
        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            eta='auto',
            per_word_topics=True,
            minimum_probability=0.01  # Lower threshold for topic assignment
        )

        topics = lda_model.print_topics(num_words=4, num_topics=num_topics)
        
        # Debug output
        print(f"Generated {num_topics} topics:")
        for topic in topics:
            print(topic)
        print(f"Total documents: {len(documents)}")
        print(f"Vocabulary size: {len(dictionary)}")
        
        return topics

    except Exception as e:
        print(f"Error processing discussion: {str(e)[:280]}")  # Truncate long error messages
        return None

if __name__ == '__main__':
    data = load_kaggle_data("./data/getting-started.json")

    for i, discussion in enumerate(data[:10]):
        print(f"\nAnalyzing discussion {i+1}: {discussion.get('title', 'No title')}")
        topics = lda_analysis(discussion)
        if topics:
            print("Detected topics:")
            for topic in topics:
                print(topic)
        else:
            print("No valid text content for topic analysis")
