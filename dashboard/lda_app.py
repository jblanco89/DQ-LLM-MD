import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import ast
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# to do: Arreglar los errores venidos del archivo de origen


# Configuración de la página
st.set_page_config(
    page_title="Análisis KL Divergencia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejor apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stMetric > label {
        font-size: 0.9rem !important;
        color: #64748b !important;
    }
    .stMetric > div {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Carga datos de ejemplo basados en tu estructura"""
    data = [
        {"title": "Machine Learning Fundamentals", "domain": "AI/ML", "kl_divergence": 0.15, "user": "alice", "votes": 45, "n_comments": 12},
        {"title": "Deep Learning Networks", "domain": "AI/ML", "kl_divergence": 0.08, "user": "bob", "votes": 67, "n_comments": 8},
        {"title": "NLP Transformers", "domain": "AI/ML", "kl_divergence": 0.32, "user": "charlie", "votes": 89, "n_comments": 15},
        {"title": "Computer Vision Applications", "domain": "AI/ML", "kl_divergence": 0.12, "user": "diana", "votes": 34, "n_comments": 6},
        {"title": "Feature Engineering Techniques", "domain": "AI/ML", "kl_divergence": 0.25, "user": "eve", "votes": 23, "n_comments": 4},
        {"title": "Reinforcement Learning", "domain": "AI/ML", "kl_divergence": 0.19, "user": "frank", "votes": 41, "n_comments": 9},
        {"title": "React Component Architecture", "domain": "Web Dev", "kl_divergence": 0.18, "user": "grace", "votes": 56, "n_comments": 9},
        {"title": "Node.js Performance", "domain": "Web Dev", "kl_divergence": 0.07, "user": "henry", "votes": 78, "n_comments": 11},
        {"title": "CSS Grid Layouts", "domain": "Web Dev", "kl_divergence": 0.28, "user": "iris", "votes": 41, "n_comments": 7},
        {"title": "API Design Patterns", "domain": "Web Dev", "kl_divergence": 0.14, "user": "jack", "votes": 52, "n_comments": 13},
        {"title": "Frontend Testing", "domain": "Web Dev", "kl_divergence": 0.21, "user": "kate", "votes": 35, "n_comments": 5},
        {"title": "Database Optimization", "domain": "Backend", "kl_divergence": 0.22, "user": "liam", "votes": 61, "n_comments": 10},
        {"title": "Microservices Architecture", "domain": "Backend", "kl_divergence": 0.11, "user": "maya", "votes": 49, "n_comments": 8},
        {"title": "Cloud Infrastructure", "domain": "Backend", "kl_divergence": 0.35, "user": "noah", "votes": 73, "n_comments": 16},
        {"title": "Docker Containerization", "domain": "Backend", "kl_divergence": 0.16, "user": "olivia", "votes": 38, "n_comments": 6},
        {"title": "Data Structures Algorithms", "domain": "CS Theory", "kl_divergence": 0.19, "user": "peter", "votes": 38, "n_comments": 5},
        {"title": "Complexity Analysis", "domain": "CS Theory", "kl_divergence": 0.13, "user": "quinn", "votes": 44, "n_comments": 7},
        {"title": "Graph Theory Applications", "domain": "CS Theory", "kl_divergence": 0.29, "user": "ruby", "votes": 31, "n_comments": 4},
        {"title": "Dynamic Programming", "domain": "CS Theory", "kl_divergence": 0.17, "user": "sam", "votes": 42, "n_comments": 8}
    ]
    
    df = pd.DataFrame(data)
    df['engagement'] = df['votes'] + df['n_comments'] * 2
    df['content_type'] = df['kl_divergence'].apply(
        lambda x: 'Típico' if x < 0.15 else 'Moderado' if x < 0.25 else 'Atípico'
    )
    return df

@st.cache_data
def load_real_data(file_path: str):
    """Carga datos reales desde tu archivo Excel"""
    try:
        df = pd.read_excel(file_path)
        df['engagement'] = df['votes'] + df['n_comments'] * 2
        df['content_type'] = df['kl_divergence'].apply(
            lambda x: 'Típico' if x < 0.15 else 'Moderado' if x < 0.25 else 'Atípico'
        )
        return df
    except Exception as e:
        st.error(f"Error cargando el archivo: {str(e)}")
        return None

def calculate_domain_stats(df):
    """Calcula estadísticas por dominio"""
    stats = []
    for domain in df['domain'].unique():
        domain_data = df[df['domain'] == domain]
        
        stat = {
            'domain': domain,
            'count': len(domain_data),
            'avg_kl': domain_data['kl_divergence'].mean(),
            'std_kl': domain_data['kl_divergence'].std(),
            'min_kl': domain_data['kl_divergence'].min(),
            'max_kl': domain_data['kl_divergence'].max(),
            'avg_engagement': domain_data['engagement'].mean(),
            'typical_count': len(domain_data[domain_data['kl_divergence'] < 0.15]),
            'atypical_count': len(domain_data[domain_data['kl_divergence'] >= 0.25])
        }
        stats.append(stat)
    
    return pd.DataFrame(stats)

def create_scatter_plot(df, x_col, y_col, color_col, title):
    """Crea gráfico de dispersión con colores por dominio"""
    df['engagement'] = df['votes'] + df['n_comments'] * 2
    df['engagement'] = df['engagement'].apply(lambda x: max(x, 0))
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        size='engagement',
        hover_data=['title', 'user', 'votes', 'n_comments'],
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font_size=16,
        height=500
    )
    
    fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
    
    return fig

def create_distribution_plot(df):
    """Crea histograma de distribución KL"""
    fig = px.histogram(
        df, 
        x='kl_divergence', 
        color='content_type',
        nbins=20,
        title='Distribución de Divergencia KL por Tipo de Contenido',
        color_discrete_map={
            'Típico': '#10b981',
            'Moderado': '#f59e0b', 
            'Atípico': '#ef4444'
        }
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        xaxis_title="Divergencia KL",
        yaxis_title="Frecuencia"
    )
    
    return fig

def create_box_plot(df):
    """Crea box plot por dominio"""
    fig = px.box(
        df, 
        x='domain', 
        y='kl_divergence',
        color='domain',
        title='Distribución de KL por Dominio',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        showlegend=False
    )
    
    return fig

def create_correlation_heatmap(df):
    """Crea mapa de correlación"""
    numeric_cols = ['kl_divergence', 'votes', 'n_comments', 'engagement']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matriz de Correlación",
        color_continuous_scale="RdBu_r"
    )
    
    fig.update_layout(height=400)
    return fig

def create_radar_chart(domain_stats):
    """Crea gráfico radar de estadísticas por dominio"""
    fig = go.Figure()
    
    # Normalizar valores para el radar
    metrics = ['avg_kl', 'avg_engagement', 'count']
    normalized_stats = domain_stats.copy()
    
    for metric in metrics:
        max_val = normalized_stats[metric].max()
        normalized_stats[f'{metric}_norm'] = normalized_stats[metric] / max_val
    
    for _, row in normalized_stats.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['avg_kl_norm'], row['avg_engagement_norm'], row['count_norm']],
            theta=['KL Promedio', 'Engagement Promedio', 'Cantidad'],
            fill='toself',
            name=row['domain']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Comparación de Dominios (Normalizado)",
        height=500
    )
    
    return fig

def create_engagement_vs_kl_plot(df):
    """Análisis de engagement vs KL divergence"""
    fig = px.scatter(
        df,
        x='kl_divergence',
        y='engagement',
        color='domain',
        size='votes',
        hover_data=['title', 'user'],
        title='Relación entre Divergencia KL y Engagement',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Añadir línea de tendencia
    z = np.polyfit(df['kl_divergence'], df['engagement'], 1)
    p = np.poly1d(z)
    fig.add_traces(go.Scatter(
        x=df['kl_divergence'].sort_values(),
        y=p(df['kl_divergence'].sort_values()),
        mode='lines',
        name='Tendencia',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">📊 Análisis de Divergencia Kullback-Leibler</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Medición de atipicidad de contenido respecto a dominios de conocimiento</div>', unsafe_allow_html=True)
    
    # Sidebar para controles
    st.sidebar.title("⚙️ Controles")
    
    # Opción para cargar datos
    data_source = st.sidebar.radio(
        "Fuente de datos:",
        ["Datos de ejemplo", "Cargar archivo Excel"]
    )
    
    if data_source == "Cargar archivo Excel":
        uploaded_file = st.sidebar.file_uploader(
            "Cargar archivo Excel", 
            type=['xlsx', 'xls']
        )
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.sidebar.success("Archivo cargado exitosamente!")
        else:
            st.warning("Por favor, carga un archivo Excel para continuar.")
            st.stop()
    else:
        df = load_sample_data()
    
    # Filtros
    st.sidebar.subheader("🔍 Filtros")
    
    domains = ['Todos'] + list(df['domain'].unique())
    selected_domain = st.sidebar.selectbox("Dominio:", domains)
    
    kl_range = st.sidebar.slider(
        "Rango de KL Divergencia:",
        float(df['kl_divergence'].min()),
        float(df['kl_divergence'].max()),
        (float(df['kl_divergence'].min()), float(df['kl_divergence'].max())),
        step=0.01
    )
    
    # Aplicar filtros
    filtered_df = df.copy()
    if selected_domain != 'Todos':
        filtered_df = filtered_df[filtered_df['domain'] == selected_domain]
    
    filtered_df = filtered_df[
        (filtered_df['kl_divergence'] >= kl_range[0]) & 
        (filtered_df['kl_divergence'] <= kl_range[1])
    ]
    
    # Métricas principales
    st.subheader("📈 Métricas Generales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_kl = filtered_df['kl_divergence'].mean()
        st.metric(
            "KL Promedio", 
            f"{avg_kl:.3f}",
            delta=f"{(avg_kl - df['kl_divergence'].mean()):.3f}"
        )
    
    with col2:
        typical_count = len(filtered_df[filtered_df['kl_divergence'] < 0.15])
        st.metric("Contenido Típico", typical_count)
    
    with col3:
        atypical_count = len(filtered_df[filtered_df['kl_divergence'] >= 0.25])
        st.metric("Contenido Atípico", atypical_count)
    
    with col4:
        total_domains = filtered_df['domain'].nunique()
        st.metric("Dominios Activos", total_domains)
    
    # Tabs para diferentes visualizaciones
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Análisis Principal", 
        "📈 Distribuciones", 
        "🎯 Correlaciones", 
        "📋 Estadísticas", 
        "🔍 Análisis Detallado"
    ])
    
    with tab1:
        st.subheader("Análisis de Dispersión y Engagement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scatter_fig = create_scatter_plot(
                filtered_df, 'votes', 'kl_divergence', 'domain',
                'Votos vs Divergencia KL'
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        with col2:
            engagement_fig = create_engagement_vs_kl_plot(filtered_df)
            st.plotly_chart(engagement_fig, use_container_width=True)
    
    with tab2:
        st.subheader("Distribuciones y Patrones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dist_fig = create_distribution_plot(filtered_df)
            st.plotly_chart(dist_fig, use_container_width=True)
        
        with col2:
            box_fig = create_box_plot(filtered_df)
            st.plotly_chart(box_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Análisis de Correlaciones")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            corr_fig = create_correlation_heatmap(filtered_df)
            st.plotly_chart(corr_fig, use_container_width=True)
        
        with col2:
            # Estadísticas de correlación
            st.write("**Correlaciones Significativas:**")
            corr_kl_votes = filtered_df['kl_divergence'].corr(filtered_df['votes'])
            corr_kl_comments = filtered_df['kl_divergence'].corr(filtered_df['n_comments'])
            corr_kl_engagement = filtered_df['kl_divergence'].corr(filtered_df['engagement'])
            
            st.metric("KL vs Votos", f"{corr_kl_votes:.3f}")
            st.metric("KL vs Comentarios", f"{corr_kl_comments:.3f}")
            st.metric("KL vs Engagement", f"{corr_kl_engagement:.3f}")
    
    with tab4:
        st.subheader("Estadísticas por Dominio")
        
        domain_stats = calculate_domain_stats(filtered_df)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(
                domain_stats.round(3),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            radar_fig = create_radar_chart(domain_stats)
            st.plotly_chart(radar_fig, use_container_width=True)
    
    with tab5:
        st.subheader("Análisis Detallado por Contenido")
        
        # Mostrar contenido más atípico
        st.write("**Top 10 Contenido Más Atípico:**")
        top_atypical = filtered_df.nlargest(10, 'kl_divergence')[
            ['title', 'domain', 'kl_divergence', 'votes', 'n_comments', 'user']
        ]
        st.dataframe(top_atypical, use_container_width=True, hide_index=True)
        
        # Mostrar contenido más típico
        st.write("**Top 10 Contenido Más Típico:**")
        top_typical = filtered_df.nsmallest(10, 'kl_divergence')[
            ['title', 'domain', 'kl_divergence', 'votes', 'n_comments', 'user']
        ]
        st.dataframe(top_typical, use_container_width=True, hide_index=True)
        
        # Análisis estadístico
        st.write("**Análisis Estadístico:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Distribución KL:**")
            st.write(f"Media: {filtered_df['kl_divergence'].mean():.3f}")
            st.write(f"Mediana: {filtered_df['kl_divergence'].median():.3f}")
            st.write(f"Desv. Estándar: {filtered_df['kl_divergence'].std():.3f}")
        
        with col2:
            st.write("**Percentiles:**")
            st.write(f"P25: {filtered_df['kl_divergence'].quantile(0.25):.3f}")
            st.write(f"P50: {filtered_df['kl_divergence'].quantile(0.50):.3f}")
            st.write(f"P75: {filtered_df['kl_divergence'].quantile(0.75):.3f}")
            st.write(f"P90: {filtered_df['kl_divergence'].quantile(0.90):.3f}")
        
        with col3:
            st.write("**Clasificación:**")
            typical_pct = (len(filtered_df[filtered_df['kl_divergence'] < 0.15]) / len(filtered_df)) * 100
            moderate_pct = (len(filtered_df[(filtered_df['kl_divergence'] >= 0.15) & (filtered_df['kl_divergence'] < 0.25)]) / len(filtered_df)) * 100
            atypical_pct = (len(filtered_df[filtered_df['kl_divergence'] >= 0.25]) / len(filtered_df)) * 100
            
            st.write(f"Típico: {typical_pct:.1f}%")
            st.write(f"Moderado: {moderate_pct:.1f}%")
            st.write(f"Atípico: {atypical_pct:.1f}%")
    
    # Footer con botón de descarga
    st.sidebar.markdown("---")
    if st.sidebar.button("📁 Descargar Datos Filtrados"):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="💾 Descargar CSV",
            data=csv,
            file_name=f"kl_analysis_{selected_domain.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()