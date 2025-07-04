from src.config.config import EXPERTISE_RANKS, DOMAINS_TAGS, MEDAL_BOOST
from src.preprocess import DataProcessor
from src.pagerank_analysis import PageRankAnalysis
from src.visualize import Visualize

def run():
    # Step 1: Process files and save results
    # data_dir = "./data/"
    # processed_output = "./src/results/processed_data.xlsx"
 

    # processor = DataProcessor(DOMAINS_TAGS, EXPERTISE_RANKS)
    # print("Processing files...")
    # dfs = processor.process_files(data_dir)
    # processor.save_to_excel(dfs, processed_output)
    # print(f"Data processed and saved to {processed_output}")

    # # Step 2: Run PageRank analysis
    # ranked_output = "./src/results/ranked_users.xlsx"
    # analysis = PageRankAnalysis(MEDAL_BOOST, EXPERTISE_RANKS)
    # print("Running PageRank analysis...")
    # rankings_df = analysis.rank_users(processed_output)
    # print("Top 10 users:")
    # print(rankings_df.head(10))
    # rankings_df.to_excel(ranked_output, index=False)
    # print(f"Ranked users saved to {ranked_output}")

    # Step 3: Visualize results
    viz = Visualize(DOMAINS_TAGS)
    print("Generating visualizations...")
    dir_kl_div = "./src/results/kl_content_vs_domain.xlsx"
    scored_output = "./src/results/data_deliq_scores.xlsx"

    # viz.upset_plot(processed_output, sheet_name="Discussions")
    # viz.plot_heatmap(
    #     ranked_output,
    #     index_name="expertise",
    #     columns_name="medal",
    #     values_name="influence_score",
    #     title="Average Influence Score by Expertise and Medal"
    # )
    # viz.plot_heatmap(
    #     processed_output,
    #     index_name="expertise",
    #     columns_name="medal",
    #     values_name="votes",
    #     title="Average Votes by Expertise and Medal"
    # )
    # viz.plot_wordcloud(
    #     processed_output,
    #     forum_name="getting-started",
    #     title="Word Cloud of Topics in Getting Started forum"
    # )
    # viz.plot_wordcloud(
    #     processed_output, 
    #     forum_name='accomplishments', 
    #     title="Word Cloud of Topics in Accomplishments forum"
    # )
    # viz.plot_wordcloud(
    #     processed_output, 
    #     forum_name='competition-hosting', 
    #     title="Word Cloud of Topics in Competition Hosting forum"
    # )

    # viz.plot_kl_divergence(
    #     dir_kl_div,
    #     title="KL Divergence Results"
    # )
    viz.plot_histogram(scored_output, 
                       column_name="score", 
                       title="Histograma de Puntuación de calidad Argumentativa (AQuA)")
    
    viz.plot_adapter_box(scored_output)
    
    print("Visualizations generated.")

if __name__ == "__main__":
    run()
