import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime
from typing import List, Dict
# from config.config import EXPERTISE_RANKS, MEDAL_BOOST
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class PageRankAnalysis:
    # def __init__(self):
    #     pass

    def __init__(self, medal_boost: Dict[str, List[str]], expertise_ranks: Dict[str, int]):
        self.medal_boost = medal_boost
        self.expertise_ranks = expertise_ranks
        
    
    def load_and_process_data(self, excel_path):
        """Load and preprocess the Excel data."""
        
        discussions_df = pd.read_excel(excel_path, sheet_name='Discussions')
        comments_df = pd.read_excel(excel_path, sheet_name='Comments')
        
        comments_df['is_appreciation'] = comments_df['is_appreciation'] == 1
        comments_df['is_deleted'] = comments_df['is_deleted'] == 1
        
        discussions_df['datetime'] = pd.to_datetime(discussions_df['datetime'])
        comments_df['datetime'] = pd.to_datetime(comments_df['datetime'])
        
        return discussions_df, comments_df

    def build_weighted_graph(self, discussions_df, comments_df):
        """Construct a directed graph with weighted edges based on forum interactions."""
        G = nx.DiGraph()
        
        # Add discussion authors as nodes
        for _, row in discussions_df.iterrows():
            G.add_node(row['user'], 
                     medal=row['medal'], 
                     expertise=row['expertise'],
                     n_posts=1,
                     total_post_votes=row['votes'],
                     n_comments=row['n_comments'],
                     n_appreciations=row['n_appreciation_comments'],)
        
        # Process comments to build edges
        for _, comment in comments_df.iterrows():
            if comment['is_deleted']:
                continue  # Skip deleted comments
                
            # Get discussion info
            discussion = discussions_df[discussions_df['id'] == comment['discussion_id']].iloc[0]
            
            commenter = comment['user']
            post_author = discussion['user']
            
            # Add commenter as node if not exists
            if commenter not in G:
                G.add_node(commenter, 
                          medal=comment['medal'], 
                          expertise=comment['expertise'],
                          n_posts=0,
                          total_comment_votes=comment['votes'],
                          is_appreciation=comment['is_appreciation'])
            
            # Calculate edge weight from commenter to post author
            base_weight = 1.0  # Base weight for the edge
            votes_weight = comment['votes']
            appreciation_boost = 1 if comment['is_appreciation'] else 0.01

            # Medal boosts for both participants
            commenter_medal_boost = self.medal_boost.get(comment['medal'], 0)
            poster_medal_boost = self.medal_boost.get(discussion['medal'], 0)
            combined_medal_boost = (commenter_medal_boost + poster_medal_boost) / 2

            # Expertise boosts
            commenter_expertise_boost = self.expertise_ranks.get(comment['expertise'], 0)
            poster_expertise_boost = self.expertise_ranks.get(discussion['expertise'], 0)
            combined_expertise_boost = (commenter_expertise_boost + poster_expertise_boost) / 2

            # Recency factor
            days_old = (pd.to_datetime('2024-03-27') - comment['datetime'].replace(tzinfo=None)).days
            recency_factor = max(0.1, 1 - (days_old / 365))
            # recency_factor = 0.5  # Placeholder for recency factor

            # Final edge weight

            edge_weight = (base_weight + votes_weight) * appreciation_boost * combined_medal_boost * combined_expertise_boost * recency_factor
            # edge_weight = 1 / (1 + math.exp(-raw_edge_weight))
            # Add the weighted edge
            if G.has_edge(commenter, post_author):
                # If edge exists, sum the weights
                G[commenter][post_author]['weight'] += edge_weight
            else:
                G.add_edge(commenter, post_author, weight=edge_weight)
        
        return G

    def visualize_graph_improved(self, G, top_n: int = 20, weight_threshold: float = 0.1, layout_algorithm='spring'):
        """
        Visualización mejorada del grafo con mejor distribución y claridad visual.
        
        Parameters:
        - G: NetworkX graph
        - top_n: número de nodos más importantes a mostrar
        - weight_threshold: umbral mínimo de peso para mostrar aristas
        - layout_algorithm: 'spring', 'kamada_kawai', 'circular', 'spectral'
        """
        
        # Configuración de colores mejorada
        medal_colors = {
            'bronze': '#CD7F32',
            'silver': '#C0C0C0', 
            'gold': '#FFD700',
            None: '#87CEEB',        # Sky blue para usuarios sin medalla
            'nan': '#87CEEB',
            float('nan'): '#87CEEB',
            'no_medal': '#87CEEB'
        }
        
        def get_medal_color(medal):
            if pd.isna(medal) or medal == '' or str(medal).lower() in ['nan', 'none']:
                return medal_colors[None]
            return medal_colors.get(str(medal).lower(), medal_colors[None])
        
        # 1. Filtrar y seleccionar nodos importantes
        if len(G.nodes()) == 0:
            print("El grafo está vacío")
            return
        
        # Calcular métricas de centralidad
        try:
            pagerank_scores = nx.pagerank(G, weight='weight', max_iter=100)
            # Considerar estas métricas para trabajo futuro

            # betweenness = nx.betweenness_centrality(G, weight='weight')
            # closeness = nx.closeness_centrality(G, distance='weight')
        except:
            pagerank_scores = {node: 1/len(G.nodes()) for node in G.nodes()}
            # Considerar estas métricas para trabajo futuro

            # betweenness = {node: 0 for node in G.nodes()}
            # closeness = {node: 0 for node in G.nodes()}
        
        # Combinar métricas para seleccionar nodos importantes
        node_importance = {}
        for node in G.nodes():
            importance = (pagerank_scores.get(node, 0))
            # Considerar estas métricas para trabajo futuro

            # importance = (
            #     pagerank_scores.get(node, 0) * 0.5 +
            #     betweenness.get(node, 0) * 0.3 +
            #     closeness.get(node, 0) * 0.2 +
            #     G.degree(node, weight='weight') * 0.001  # Normalizar grado
            # )
            node_importance[node] = importance
        
        # Seleccionar top nodos
        top_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        selected_nodes = [node for node, _ in top_nodes]
        
        # 2. Filtrar aristas por peso y crear subgrafo
        filtered_edges = []
        for u, v, data in G.edges(data=True):
            if (u in selected_nodes and v in selected_nodes and 
                data.get('weight', 0) >= weight_threshold):
                filtered_edges.append((u, v))
        
        G_filtered = G.subgraph(selected_nodes).copy()
        
        if len(G_filtered.nodes()) == 0:
            print("No hay nodos que cumplan los criterios de filtrado")
            return
        
        # 3. Seleccionar algoritmo de layout
        layouts = {
            'spring': lambda g: nx.spring_layout(g, k=3, iterations=50, weight='weight', seed=42),
            'kamada_kawai': lambda g: nx.kamada_kawai_layout(g, weight='weight'),
            'circular': lambda g: nx.circular_layout(g),
            # 'spectral': lambda g: nx.spectral_layout(g, weight='weight'),
            'shell': lambda g: nx.shell_layout(g)
        }
        
        try:
            pos = layouts.get(layout_algorithm, layouts['spring'])(G_filtered)
        except:
            pos = nx.spring_layout(G_filtered, k=2, iterations=30, seed=42)
        
        # 4. Crear visualización mejorada
        plt.figure(figsize=(16, 12))
        plt.clf()
        
        # Calcular tamaños de nodos basados en importancia
        node_sizes = []
        min_size, max_size = 400, 3000
        importance_values = [node_importance[node] for node in G_filtered.nodes()]
        
        if len(set(importance_values)) > 1:
            normalized_importance = (np.array(importance_values) - min(importance_values)) / (max(importance_values) - min(importance_values))
            node_sizes = min_size + normalized_importance * (max_size - min_size)
        else:
            node_sizes = [min_size] * len(G_filtered.nodes())
        
        # Dibujar nodos con bordes
        node_colors = [get_medal_color(G_filtered.nodes[node].get('medal')) for node in G_filtered.nodes()]
        
        nx.draw_networkx_nodes(
            G_filtered, pos,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors='black',
            linewidths=1.5,
            alpha=0.9
        )
        
        # 5. Dibujar aristas con grosor proporcional al peso
        edges = G_filtered.edges(data=True)
        if len(edges) > 0:
            weights = [data.get('weight', 1) for _, _, data in edges]
            max_weight = max(weights) if weights else 1
            min_weight = min(weights) if weights else 1

            # Normalizar grosores de aristas
            if max_weight > min_weight:
                normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
                edge_widths = [0.5 + nw * 4 for nw in normalized_weights]  # Entre 0.5 y 4.5
            else:
                edge_widths = [1.0] * len(weights)
            
            nx.draw_networkx_edges(
                G_filtered, pos,
                width=edge_widths,
                alpha=0.7,
                edge_color='gray',
                arrows=True,
                arrowsize=20,
                arrowstyle='->'
            )
            
            # Mostrar pesos solo en aristas importantes (top 10)
            sorted_edges = sorted(edges, key=lambda x: x[2].get('weight', 0), reverse=True)
            top_edges = sorted_edges[:min(10, len(sorted_edges))]
            
            edge_labels = {}
            for u, v, data in top_edges:
                weight = data.get('weight', 0)
                if weight >= 0.1:
                    edge_labels[(u, v)] = f"{weight:.1f}"
            
            if edge_labels:
                nx.draw_networkx_edge_labels(
                    G_filtered, pos, 
                    edge_labels=edge_labels, 
                    font_size=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
                )
        
        # 6. Etiquetas de nodos mejoradas
        labels = {}
        for node in G_filtered.nodes():
            # Mostrar solo primeros 10 caracteres del nombre
            short_name = str(node)[:10] + "..." if len(str(node)) > 10 else str(node)
            labels[node] = short_name
        
        nx.draw_networkx_labels(
            G_filtered, pos, 
            labels=labels,
            font_size=10,
            font_weight='bold',
            font_color='black'
        )
        
        # 7. Leyenda mejorada
        legend_elements = []
        for medal, color in medal_colors.items():
            if medal and str(medal) != 'nan':
                label = medal.title() if medal != None else 'Sin medalla'
                legend_elements.append(plt.scatter([], [], c=color, s=100, label=label, edgecolors='black'))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # 8. Título y configuración final
        plt.title(f'Red de Influencia - Top {len(G_filtered.nodes())} Usuarios\n'
                  f'Algoritmo: {layout_algorithm.title()}, Umbral: {weight_threshold}', 
                  fontsize=16, fontweight='bold', pad=20)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Guardar con alta resolución
        plt.savefig(f'./src/results/network_graph_{layout_algorithm}.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # 9. Estadísticas del grafo
        print(f"\n=== Estadísticas del Grafo ===")
        print(f"Nodos mostrados: {len(G_filtered.nodes())}")
        print(f"Aristas mostradas: {len(G_filtered.edges())}")
        print(f"Densidad: {nx.density(G_filtered):.3f}")
        print(f"Componentes conectadas: {nx.number_weakly_connected_components(G_filtered)}")
        

        # Esto pude ser útil para un trabajo posterior y prospectiva futura

        # # Top 5 usuarios por importancia
        # print(f"\n=== Top 5 Usuarios por Importancia ===")
        # for i, (node, importance) in enumerate(top_nodes[:5], 1):
        #     medal = G.nodes[node].get('medal', 'Sin medalla')
        #     print(f"{i}. {node} - Importancia: {importance:.4f} - Medalla: {medal}")

    def calculate_influence_scores(self, discussions_df, comments_df):
        """
        Versión mejorada de calculate_influence_scores con mejor visualización.
        """
        # Tu código existente para construir el grafo
        G = self.build_weighted_graph(discussions_df, comments_df)
        
        # Calcular PageRank
        pagerank_scores = nx.pagerank(G, weight='weight')
        
        # Tu código existente para métricas de usuario...
        user_metrics = {}
        
        for user in G.nodes():
            user_metrics[user] = {
                'total_post_votes': G.nodes[user].get('total_post_votes', 0),
                'comment_count': G.nodes[user].get('n_comments', 0),
                # 'total_comment_votes': G.nodes[user].get('total_comment_votes', 0),
                'received_appreciations': G.nodes[user].get('n_appreciations', 0),
                'medal': G.nodes[user].get('medal'),
                'expertise': G.nodes[user].get('expertise', 'Novice'),
                'edge_count': G.in_degree(user)
            }
        
        # Calcular scores finales
        influence_scores = {}
        for user, metrics in user_metrics.items():
            score = pagerank_scores.get(user, 0)
            influence_scores[user] = score
        
        # VISUALIZACIONES MEJORADAS
        print("Generando visualización estática mejorada...")
        self.visualize_graph_improved(G, top_n=15, weight_threshold=0.5, layout_algorithm='spring')
        
        # print("Generando visualización interactiva...")
        # self.visualize_graph_interactive(G, top_n=20, weight_threshold=0.3)
        
        # También probar diferentes layouts
        for layout in ['kamada_kawai', 'circular', 'shell']:
            try:
                print(f"Generando visualización con layout {layout}...")
                self.visualize_graph_improved(G, top_n=15, weight_threshold=0.5, layout_algorithm=layout)
            except Exception as e:
                print(f"Error con layout {layout}: {e}")
        
        return influence_scores, user_metrics

    def rank_users(self, excel_path):
        """Main function to rank users by influence."""
        # Load and process data
        discussions_df, comments_df = self.load_and_process_data(excel_path)
        
        # Calculate influence scores
        influence_scores, user_metrics = self.calculate_influence_scores(discussions_df, comments_df)
        
        # Sort users by score
        ranked_users = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare results with all metrics
        results = []
        for rank, (user, score) in enumerate(ranked_users, 1):
            metrics = user_metrics[user]
            results.append({
                'rank': rank,
                'username': user,
                'influence_score': round(score, 6),
                'edge_count': metrics['edge_count'],
                'medal': metrics['medal'],
                'expertise': metrics['expertise'],
                'received_post_votes': round(metrics['total_post_votes']),
                'comment_count': metrics['comment_count'],
                # 'total_comment_votes': round(metrics['total_comment_votes']),
                # 'avg_comment_votes': metrics['total_comment_votes'] / metrics['comment_count'] if metrics['comment_count'] > 0 else 0,
                'received_appreciations': metrics['received_appreciations']
            })
        df = pd.DataFrame(results)
        return df

# if __name__ == "__main__":
#     excel_path = './src/results/processed_data.xlsx' 
#     analysis = PageRankAnalysis()
#     rankings_df = analysis.rank_users(excel_path)
#     print(rankings_df.head(10))  # Display top 10 users
#     rankings_df.to_excel('./src/results/ranked_users.xlsx', index=False)
