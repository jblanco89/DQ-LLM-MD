# to do: include the PageRank graph visualization
# to do: innclude a Venn diagram of the domains of users (KL/LDA analysis) [Done with UpSet]
# to do: some exploratory analysis of the data (most interesting and valuable visualizations)
# to do: include a word cloud of the most relevant words in the topics
# to do: include a heatmap of thematic dispersion by domain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet
# from wordcloud import WordCloud
from config.config import DOMAINS_TAGS

def upset_plot(directory: str = "src/results/processed_data.xlsx", sheet_name: str = "Discussions"):
    """
    Create an UpSet plot to visualize the distribution of discussions across domains.
    """
    
    df = pd.read_excel(directory, sheet_name=sheet_name)
    data = df.copy()
    data = data[["user", "domains"]]
    list_domains = list(DOMAINS_TAGS.keys())
    list_domains = list_domains + ["General"]

    # Crear matriz de pertenencia
    for domain in list_domains:
        data[domain] = data['domains'].apply(lambda x: domain in x)

    # Preparar datos para UpSet
    upset_data = data.set_index(list_domains).groupby(level=list_domains).size()

    # Leer la documentación de UpSet para más opciones de visualización
    UpSet(upset_data, subset_size='sum', show_counts=True).plot()
    

    plt.savefig("src/results/upset_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_heatmap(data: pd.DataFrame, title: str = "Heatmap"):
    """
    Create a heatmap to visualize the distribution of discussions across domains.
    """
    # Assuming 'data' is a DataFrame with domain distributions
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Intensity')
    plt.xlabel('Domains')
    plt.ylabel('Discussions')
    plt.xticks(range(len(data.columns)), data.columns, rotation=45)
    plt.yticks(range(len(data.index)), data.index)
    plt.tight_layout()
    plt.show()


def plot_wordcloud(data: pd.DataFrame, title: str = "Word Cloud"):
    """
    Create a word cloud to visualize the most relevant words in the topics.
    """
    

    text = ' '.join(data['terms'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()  


def plot_kl_divergence(data: pd.DataFrame, title: str = "KL Divergence"):
    """
    Create a plot to visualize KL divergence results.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(data['title'], data['kl_divergence'])
    plt.xlabel('Discussions')
    plt.ylabel('KL Divergence')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    dir_useplot = "src/results/processed_data.xlsx"
    upset_plot(dir_useplot)
    # df = pd.read_excel("", sheet_name="Discussions")
    # plot_heatmap(df, title="Heatmap of Discussions")
    # plot_wordcloud(df, title="Word Cloud of Topics")
    # plot_kl_divergence(df, title="KL Divergence Results")
        

