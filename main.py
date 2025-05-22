from src.config.config import EXPERTISE_RANKS, DOMAINS_TAGS
from src.preprocess import DataProcessor, TextCleaner, DomainClassifier, StructuralFeatureExtractor

# to do: import visualization functions as Class methods
# to do: import LDA analysis functions as Class methods
# to do: import structural analysis (pageRank) functions as Class methods


if __name__ == "__main__":
    data_dir = "./data/"
    output_path = "./src/results/processed_data.xlsx"    
    # Initialize the processor with your config
    processor = DataProcessor(DOMAINS_TAGS, EXPERTISE_RANKS)
    # Process files
    dfs = processor.process_files(data_dir)
    # Save results
    processor.save_to_excel(dfs, output_path)