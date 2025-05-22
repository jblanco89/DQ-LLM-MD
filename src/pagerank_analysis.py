import pandas as pd
import networkx as nx
from datetime import datetime
from config.config import EXPERTISE_RANKS, MEDAL_BOOST
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# to do: check the assigned values for weights in influence score calculation (see lines 73 and 167 and onwards) [Done]
# to do: check the recency factor (see line 63 and 64)
# to do: try to understand and interpret the graph (see line 118 and onwards)
# to do: check the edge color and edge text and edge visualization


def load_and_process_data(excel_path):
    """Load and preprocess the Excel data."""
    
    discussions_df = pd.read_excel(excel_path, sheet_name='Discussions')
    comments_df = pd.read_excel(excel_path, sheet_name='Comments')
    
    comments_df['is_appreciation'] = comments_df['is_appreciation'] == 1
    comments_df['is_deleted'] = comments_df['is_deleted'] == 1
    
    discussions_df['datetime'] = pd.to_datetime(discussions_df['datetime'])
    comments_df['datetime'] = pd.to_datetime(comments_df['datetime'])
    
    return discussions_df, comments_df

def build_weighted_graph(discussions_df, comments_df):
    """Construct a directed graph with weighted edges based on forum interactions."""
    G = nx.DiGraph()
    
    # Add discussion authors as nodes
    for _, row in discussions_df.iterrows():
        G.add_node(row['user'], 
                 medal=row['medal'], 
                 expertise=row['expertise'],
                 n_posts=1,
                 total_post_votes=row['votes'])
    
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
                      total_post_votes=0)
        
        # Calculate edge weight from commenter to post author
        base_weight = 1.0
        votes_weight = comment['votes'] * 0.5
        appreciation_boost = 1.0 if comment['is_appreciation'] else 0
        
        # Medal boosts for both participants
        commenter_medal_boost = MEDAL_BOOST.get(comment['medal'], 0)
        poster_medal_boost = MEDAL_BOOST.get(discussion['medal'], 0)
        combined_medal_boost = (commenter_medal_boost + poster_medal_boost) / 2
        
        # Expertise boosts
        commenter_expertise_boost = EXPERTISE_RANKS.get(comment['expertise'], 0)
        poster_expertise_boost = EXPERTISE_RANKS.get(discussion['expertise'], 0)
        combined_expertise_boost = (commenter_expertise_boost + poster_expertise_boost) / 2
        
        # Recency factor
        # days_old = (datetime.now() - comment['datetime']).days
        # recency_factor = max(0.1, 1 - (days_old / 365))
        recency_factor = 1.0  # Placeholder for recency factor
        
        # Final edge weight
        edge_weight = (base_weight + votes_weight) * appreciation_boost * combined_medal_boost * combined_expertise_boost * recency_factor
        
        # Add the weighted edge
        if G.has_edge(commenter, post_author):
            # If edge exists, sum the weights
            G[commenter][post_author]['weight'] += edge_weight
        else:
            G.add_edge(commenter, post_author, weight=edge_weight)
    
    return G

def visualize_graph(G, top_n=10):
    """Create both interactive and static visualizations of the graph"""
    # Medal colors with NaN handling
    medal_colors = {
        'bronze': '#cd7f32',
        'silver': '#c0c0c0',
        'gold': '#ffd700',
        None: '#888888',
        'nan': '#888888',  # Handle string 'nan'
        float('nan'): '#888888'  # Handle numpy/pandas NaN
    }

    # Interactive visualization with Plotly
    sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:top_n]
    G = G.subgraph(sorted_nodes)
    pos = nx.spring_layout(G, k=0.35, iterations=20, scale=3)
    
    # Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        medal = G.nodes[node].get('medal')
        expertise = G.nodes[node].get('expertise', 'Novice')
        node_text.append(f"{node}<br>Medal: {medal if pd.notna(medal) else 'None'}<br>Expertise: {expertise}")

    # Safe medal color lookup
    def get_medal_color(medal):
        try:
            if pd.isna(medal):
                return medal_colors[None]
            return medal_colors.get(str(medal).lower(), medal_colors[None])
        except:
            return medal_colors[None]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node[:10] for node in G.nodes()],
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=[get_medal_color(G.nodes[node].get('medal')) for node in G.nodes()],
            size=[min(G.degree(node)*3, 50) for node in G.nodes()],  # Cap size at 50
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    fig.show()

    # Static visualization with matplotlib
    plt.figure(figsize=(12,12))
    nx.draw_networkx_nodes(G, pos, 
                         node_size=[min(G.degree(n)*50, 1000) for n in G.nodes()],  # Cap size at 1000
                         node_color=[get_medal_color(G.nodes[n].get('medal')) for n in G.nodes()])
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif')
    plt.axis('on')
    plt.savefig('./src/results/network_graph.png', dpi=300)
    plt.show()



def calculate_influence_scores(discussions_df, comments_df):
    """Calculate comprehensive influence scores combining PageRank with other metrics."""
    # Build the weighted graph
    G = build_weighted_graph(discussions_df, comments_df)
    
    # Run weighted PageRank
    pagerank_scores = nx.pagerank(G, weight='weight')
    
    # Calculate additional metrics
    user_metrics = {}
    
    # Initialize with graph nodes
    for user in G.nodes():
        user_metrics[user] = {
            'post_count': G.nodes[user].get('n_posts', 0),
            'total_post_votes': G.nodes[user].get('total_post_votes', 0),
            'comment_count': 0,
            'total_comment_votes': 0,
            'received_comments': 0,
            'received_appreciations': 0,
            'medal': G.nodes[user].get('medal'),
            'expertise': G.nodes[user].get('expertise', 'Novice')
        }
    
    # Count comments made by each user
    user_comments = comments_df.groupby('user').agg({
        'id': 'count',
        'votes': 'sum',
        'is_appreciation': 'sum'
    }).reset_index()
    
    for _, row in user_comments.iterrows():
        if row['user'] in user_metrics:
            user_metrics[row['user']]['comment_count'] = row['id']
            user_metrics[row['user']]['total_comment_votes'] = row['votes']
    
    # Count comments and appreciations received
    received_counts = comments_df.merge(
        discussions_df[['id', 'user']], 
        left_on='discussion_id', 
        right_on='id', 
        suffixes=('', '_post')
    )
    
    received_stats = received_counts.groupby('user_post').agg({
        'id': 'count',
        'is_appreciation': 'sum'
    }).reset_index()
    
    for _, row in received_stats.iterrows():
        if row['user_post'] in user_metrics:
            user_metrics[row['user_post']]['received_comments'] = row['id']
            user_metrics[row['user_post']]['received_appreciations'] = row['is_appreciation']
    
    
    # Calculate final influence scores
    influence_scores = {}
    for user, metrics in user_metrics.items():
        # Base score from PageRank
        score = pagerank_scores.get(user, 0)
        influence_scores[user] = score
    # Add this line at the end of your calculate_influence_scores function, before return:
    visualize_graph(G, top_n=10)
    return influence_scores, user_metrics

def rank_users(excel_path):
    """Main function to rank users by influence."""
    # Load and process data
    discussions_df, comments_df = load_and_process_data(excel_path)
    
    # Calculate influence scores
    influence_scores, user_metrics = calculate_influence_scores(discussions_df, comments_df)
    
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
            'medal': metrics['medal'],
            'expertise': metrics['expertise'],
            'post_count': metrics['post_count'],
            'avg_post_votes': round(metrics['total_post_votes'] / max(1, metrics['post_count']), 2),
            'comment_count': metrics['comment_count'],
            'avg_comment_votes': round(metrics['total_comment_votes'] / max(1, metrics['comment_count']), 2),
            'received_comments': metrics['received_comments'],
            'received_appreciations': metrics['received_appreciations']
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    excel_path = './src/results/processed_data.xlsx' 
    rankings_df = rank_users(excel_path)
    print(rankings_df.head(10))  # Display top 10 users
    rankings_df.to_excel('./src/results/ranked_users.xlsx', index=False)