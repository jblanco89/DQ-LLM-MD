# LLMs and Deliberation Quality Analysis Project

## Overview
This project analyzes Kaggle forum discussions using Natural Language Processing (NLP) techniques, focusing on structural analysis and topic modeling. The project includes tools for preprocessing text data, extracting structural features, and performing Latent Dirichlet Allocation (LDA) analysis.

## Project Structure
```
src/
├── analysis.py      # Topic modeling and text analysis
├── preprocess.py    # Data loading and feature extraction
└── visualize.py     # Data visualization (placeholder)
```

## Key Features
- Text preprocessing and cleaning
- Structural feature extraction from discussions
- Topic modeling using LDA
- Technical term detection
- Thread depth analysis
- Expertise level analysis

## Modules

### `preprocess.py`
Handles data loading and preprocessing tasks.

**Key Functions:**
- `load_kaggle_data(path: str) -> List[Dict]`: Loads Kaggle discussion data from JSON files
- `clean_text(text: str) -> str`: Normalizes text for analysis
- `extract_structural_features(discussion: Dict) -> Dict`: Extracts metrics including:
  - Thread depth
  - Average expertise level
  - Vote ratio
  - Comment counts
- `calculate_thread_depth(comments: Dict) -> int`: Recursively calculates discussion thread depth

### `analysis.py`
Implements topic modeling and text analysis.

**Key Functions:**
- `preprocess_text(text: str) -> list`: Performs text preprocessing including:
  - URL and code block removal
  - User mention handling
  - Lemmatization
  - Stopword removal
- `lda_analysis(discussion: dict) -> list`: Performs LDA topic modeling with:
  - Adaptive topic number selection
  - Technical term preservation
  - Phrase detection
  - Vocabulary filtering

## Technical Details

### Expertise Ranking System
```python
EXPERTISE_RANKS = {
    "Novice": 1,
    "Contributor": 2,
    "Expert": 3,
    "Master": 4,
    "Grandmaster": 5,
    None: 0 
}
```

### Technical Terms
Includes domain-specific terms like:
- Machine learning concepts
- Programming frameworks
- Data science terminology
- NLP-related terms

## Usage Example

```python
from src.preprocess import load_kaggle_data, extract_structural_features
from src.analysis import lda_analysis

# Load data
data = load_kaggle_data("./data/getting-started.json")

# Analyze discussions
for i, discussion in enumerate(data[:10]):
    # Extract structural features
    features = extract_structural_features(discussion)
    print(f"Discussion ID: {i+1}, Features: {features}")
    
    # Perform topic analysis
    topics = lda_analysis(discussion)
    if topics:
        print("Detected topics:")
        for topic in topics:
            print(topic)
```

## Dependencies
- NLTK
- Gensim
- NumPy
- Regular Expressions (re)

## TODO List
- [ ] Add configuration file for constants
- [ ] Implement data append functionality
- [ ] Add CSV/JSON export functionality
- [ ] Improve off-topic handling
- [ ] Consider BERTopic as an LDA alternative
- [ ] Merge preprocessing functions
- [ ] Add automatic NLTK data download
- [ ] Enhance hierarchical comment analysis

## Installation
```bash
conda create -n master_project python=3.10.16
conda activate  master_projec

pip install -r requirements.txt

python -m nltk.downloader stopwords wordnet
```

## Notes
- The project is designed to handle missing data gracefully
- Includes debug outputs for analysis steps
- Uses adaptive parameters based on data size
- Preserves technical terminology during preprocessing