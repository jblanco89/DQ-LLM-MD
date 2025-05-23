# LLMs and Deliberation Quality Analysis Project

## Overview
This project analyzes online forum discussions using Natural Language Processing (NLP) and Large Language Models (LLMs) to evaluate deliberation quality. The analysis focuses on structural features, topic modeling, and interaction patterns in Kaggle forum discussions.

## Project Structure
```
project/
├── src/
│   ├── analysis.py      # Topic modeling and text analysis
│   ├── preprocess.py    # Data loading and feature extraction
│   └── visualize.py     # Data visualization
├── data/
│   └── competition-hosting.json
├── docker/
│   └── Dockerfile
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
├── main.py
└── README.md
```

## Key Features
- Advanced text preprocessing and cleaning
- Structural feature extraction from discussions
- Topic modeling using LDA
- Technical term detection
- Thread depth and interaction analysis
- Expertise level evaluation
- Docker containerization
- Interactive visualization of discussion networks

## Dependencies
- Python 3.10.16
- NLTK
- Gensim
- NumPy
- Docker (optional)

## Installation

### Using Conda (Recommended)
```bash
# Create and activate conda environment
conda create -n master_project python=3.10.16
conda activate master_project

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader stopwords wordnet
```

### Using Docker
```bash
# Build Docker image
docker build -t deliberation-analysis .

# Run container
docker run -it deliberation-analysis
```

## Usage

### Basic Analysis
```python
from src.preprocess import load_kaggle_data, extract_structural_features
from src.analysis import lda_analysis

# Load and analyze data
data = load_kaggle_data("./data/competition-hosting.json")
results = analyze_discussions(data)
```

### Visualization
```python
from src.visualize import plot_interaction_network

# Generate interaction network visualization
plot_interaction_network(results)
```

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

## Configuration
The project uses environment variables for configuration:
- `PYTHONPATH=/app`
- `DATA_DIR=/app/data`

## Development Roadmap
- [x] Basic text preprocessing
- [x] LDA implementation
- [x] Docker support
- [x] Add configuration file for constants
- [x] Improve visualization options




## Notes
- Handles missing data gracefully
- Includes debug outputs for analysis steps
- Uses adaptive parameters based on data size
- Preserves technical terminology during preprocessing
- Docker container includes all necessary NLTK data
