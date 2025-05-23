# to do: innclude a Venn diagram of the domains of users (KL/LDA analysis) [Done with UpSet]
# to do: some exploratory analysis of the data (most interesting and valuable visualizations)
# to do: include a word cloud of the most relevant words in the topics [Done by forum]
# to do: include a heatmap of thematic dispersion by domain [Done with influence score and expertise-medal]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from upsetplot import UpSet
from wordcloud import WordCloud
from typing import Dict, List
# from config.config import DOMAINS_TAGS


class Visualize:
    def __init__(self, domain_tags: Dict[str, List[str]]):
        self.domain_tags = domain_tags

    def upset_plot(self, directory: str = "src/results/processed_data.xlsx", sheet_name: str = "Discussions"):
        """
        Create an UpSet plot to visualize the distribution of discussions across domains.
        """
        df = pd.read_excel(directory, sheet_name=sheet_name)
        data = df.copy()
        data = data[["user", "domains"]]
        list_domains = list(self.domain_tags.keys())
        list_domains = list_domains + ["General"]

        # Crear matriz de pertenencia
        for domain in list_domains:
            data[domain] = data['domains'].apply(lambda x: domain in x)

        # Preparar datos para UpSet
        upset_data = data.set_index(list_domains).groupby(level=list_domains).size()

        # Leer la documentación de UpSet para más opciones de visualización
        UpSet(upset_data, subset_size='sum', 
              show_counts=True).plot()
        plt.title("UpSet Plot of Discussions by Domain")
        plt.savefig("src/results/upset_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_heatmap(self, directory: str, 
                     index_name: str, 
                     columns_name: str, 
                     values_name: str,
                     agg_func: str = 'mean',
                     title: str = "Heatmap Plot"):
        """
        Create a heatmap to visualize the distribution of discussions across domains.
        """
        df = pd.read_excel(directory)
        df = df.copy()
        data = df[[values_name, index_name, columns_name]]

        # Pivot the data for heatmap
        heatmap_data = data.pivot_table(
            index=index_name,
            columns=columns_name,
            values=values_name,
            aggfunc=agg_func
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_data, 
            annot=True,  # Show values in cells
            fmt=".3f",  # 3 decimal places
            cmap="YlOrRd",  # Gold→Red gradient (matches medal theme)
            linewidths=0.5,
            linecolor='black',  # Black lines between cells
            cbar_kws={"label": f'{values_name}'}, # Color bar label

        )
        plt.title(title)
        plt.xlabel("Medal")
        plt.ylabel("Expertise Level")
        
        plt.savefig(f"src/results/heatmap_plot_{values_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_wordcloud(self, directory: str, forum_name: str, title: str = "Word Cloud"):
        """
        Create a word cloud to visualize the most relevant words in the topics.
        """
        df = pd.read_excel(directory)
        data = df.copy()
        data = data[data['forum'] == forum_name]
        text = ' '.join(data['content'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)

        plt.savefig("src/results/wordcloud_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()  

    def plot_kl_divergence(self, directory: str, title: str = "KL Divergence"):
        """
        Create a plot to visualize KL divergence results.
        """
        df = pd.read_excel(directory)
        data = df.copy()

        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.bar(data['domain'], data['kl_divergence'])
        plt.xlabel('Domains')
        plt.ylabel('KL Divergence')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig("src/results/kl_divergence_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == "__main__":
    viz = Visualize()
    
    # Example usage
    dir_processed_data = "src/results/processed_data.xlsx"
    dir_kl_div = "src/results/kl_content_vs_domain.xlsx"
    dir_ranked = "src/results/ranked_users.xlsx"
    
    viz.upset_plot(dir_processed_data, sheet_name="Discussions")
    viz.plot_heatmap(dir_ranked, index_name="expertise", columns_name="medal", values_name="influence_score", title="Average Influence Score by Expertise and Medal")
    viz.plot_heatmap(dir_processed_data, index_name="expertise", columns_name="medal", values_name="votes", title="Average Votes by Expertise and Medal")
    viz.plot_wordcloud(dir_processed_data, forum_name='getting-started', title="Word Cloud of Topics")
    viz.plot_kl_divergence(dir_kl_div, title="KL Divergence Results")
