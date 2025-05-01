from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import Phrases
from gensim.utils import simple_preprocess
from preprocess import clean_text
import pandas as pd
import ast


# to do: off-topic handling. Exclude them maybe by checking is_appreciation or is_deleted flags
# to do: manage hieriarchical comments as elegible for topic analysis optionally
# to do: consider to use an LDA alternative different from Bertopic but with similar performance or better. 
# to do: consider to merge here preprocessing text with preprocessing functions in preprocess.py [done]

def load_processed_data(path: str) -> list:
    """Load and normalize data from Kaggle"""
    with open(path, encoding='latin1') as f:
        data = pd.read_excel(path)
        return data.to_dict(orient='records')

# Initialize tools
lemmatizer = WordNetLemmatizer()

def get_tokens(text: str) -> list:
    """More conservative preprocessing that preserves key information"""
    if not text:
        return []
    
    text = clean_text(text)  # Clean the text first
    
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

    # Process main content with debug info
    content = discussion.get('content')
    if content:
        processed = get_tokens(content)
        if processed:
            documents.append(processed)
            print(f"\nProcessing discussion: {discussion.get('title', 'No title')}")
            print(f"Original content sample: {content[:280]}...")
            print(f"Processed tokens: {processed}")

    # Process comments
    comments = discussion.get("comments", {})
    comment_items = []
    if isinstance(comments, dict):
        comment_items = comments.values()
    elif isinstance(comments, str):
        try:
            comment_dict = ast.literal_eval(comments)
            if isinstance(comment_dict, dict):
                comment_items = comment_dict.values()
        except Exception as e:
            print(f"Error converting comments: {e}")

    for comment in comment_items:
        comment_content = comment.get('content')
        if comment_content:
            processed = get_tokens(comment_content)
            if processed:
                documents.append(processed)

    # Skip if insufficient data
    if not documents or sum(len(d) for d in documents) < 5:
        print("Insufficient data after preprocessing")
        return None

    technical_terms = {
        'feature_engineering', 'auto_ml', 'data_centric', 
        'model_centric', 'random_search', 'grid_search', 'machine_learning',
        'deep_learning', 'transfer_learning', 'reinforcement_learning',
        'probabilistic_modeling','programming', 'data_science', 'data_analysis',
        'data_visualization', 'natural_language_processing', 'computer_vision',
        'time_series_analysis', 'classification', 'pytorch', 'word-to-vecto',
        'word2vec', 'fasttext', 'bert', 'gpt', 'transformer','doc2vec','cnn','rnn'
    }
    # Replace spaces with underscores when the transformed token is in technical_terms
    documents = [
        [word.replace(' ', '_') if word.replace(' ', '_') in technical_terms else word 
         for word in doc]
        for doc in documents
    ]

    try:
        # Phrase detection
        bigram = Phrases(documents, min_count=2, threshold=5)
        trigram = Phrases(bigram[documents], min_count=2, threshold=5)
        documents_phrased = [trigram[bigram[doc]] for doc in documents]

        # Create dictionary and filter extremes based on document count
        dictionary = corpora.Dictionary(documents_phrased)
        if len(documents) < 10:
            dictionary.filter_extremes(no_below=1, no_above=0.9)
        else:
            dictionary.filter_extremes(no_below=2, no_above=0.7)

        # Ensure there are enough words
        if len(dictionary) < 5:
            dictionary.filter_extremes(no_below=1, no_above=0.9) 
            print(f"Insufficient vocabulary: {len(dictionary)} terms")
            return None

        corpus = [dictionary.doc2bow(doc) for doc in documents_phrased]

        # Adaptive topic number: at least 1 and at most 2 topics
        num_topics = min(2, max(1, len(documents) // 2))

        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            eta='auto',
            per_word_topics=True,
            minimum_probability=0.1  # Lower threshold for topic assignment
        )

        topics = lda_model.print_topics(num_words=4, num_topics=4)
        return topics

    except Exception as e:
        print(f"Error processing discussion: {str(e)[:280]}")
        return None

if __name__ == '__main__':
    data = load_processed_data("src/results/processed_data.xlsx")
    if not data:
        print("No data to process")
        exit(1)
    for i, discussion in enumerate(data[:5]):
        print(f"\nAnalyzing discussion {i+1}: {discussion.get('title', 'No title')}")
        topics = lda_analysis(discussion)
        if topics:
            print("Detected topics:")
            for topic in topics:
                print(topic)
        else:
            print("No valid text content for topic analysis")
