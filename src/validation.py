import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Importing files with features: DQ, KL, PR, user expertise and votes

df_dq = pd.read_csv('src/results/processed_data_inferenced.csv', sep='\t')
# df_dq = df_dq[['id', 'user', 'title', 'votes', 'expertise', 'score']]
df_dq = df_dq.drop(columns=['content', 'tags','datetime'])
print(df_dq.shape)

df_kl = pd.read_excel('src/results/kl_content_vs_domain.xlsx',)
df_kl = df_kl.loc[df_kl.groupby('user')['kl_divergence'].idxmin()].sort_values(by='id', ascending=True)
print(df_kl.shape)

df_pr = pd.read_excel('src/results/ranked_users.xlsx')
df_pr = df_pr[['username', 'influence_score', 'expertise', 'received_post_votes', 'rank']]
df_pr.rename(columns={'username': 'user'}, inplace=True)

merged_df = pd.merge(df_dq, df_kl[['id', 'kl_divergence']], on='id', how='left')

merged_df = pd.merge(merged_df, df_pr[['user','influence_score']], on='user', how='left')
merged_df[['kl_divergence', 'influence_score']] = merged_df[['kl_divergence', 'influence_score']].fillna(0)
# merged_df['votes'] = merged_df['votes'].astype(int)

def deliq_score(row, weights):
    return np.dot(row, weights)  # Producto punto: w1*v1 + w2*v2 + w3*v3

# Pesos iniciales (se optimizarán después)
initial_weights = np.array([0.50, 0.10, 0.40])

def calculate_deliq_score(df, weights):
    # Calcular el score de deliq
    df['deliq_score'] = deliq_score(df[['score','kl_divergence', 'influence_score']], weights)
    return df

merged_df = calculate_deliq_score(merged_df, initial_weights)
print(merged_df.columns)

expertise_order =  {"Novice": 1,
    "Contributor": 2,
    "Expert": 3,
    "Master": 4,
    "Grandmaster": 5,
    None: 0}

merged_df['expertise_ordinal'] = merged_df['expertise'].map(expertise_order).fillna(0)

merged_df.to_excel('src/results/data_deliq_scores.xlsx', index=False)

array_data = merged_df.to_dict(orient='records')

def dbscan_cluster_validation(results, eps=0.5, min_samples=10, plot=True, show_noise=False):
    """
    Agrupa comentarios usando DBSCAN y valida los clusters
    
    Args:
        show_noise: Si es False, los outliers (ruido) no se mostrarán en la gráfica
    """
    # Extraer y escalar características
    df = pd.DataFrame(results)
    # features = df[['deliq_score', 'influence_score', 'kl_divergence', 'score']].values
    features = StandardScaler().fit_transform(df[['deliq_score', 'influence_score', 'kl_divergence', 'score']])

    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features)
    
    # Calcular métricas
    metrics = {}
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    
    if n_clusters > 1:
        non_noise = clusters != -1
        if sum(non_noise) > 1:
            metrics = {
                "silhouette_score": silhouette_score(features[non_noise], clusters[non_noise]),
                "calinski_harabasz_score": calinski_harabasz_score(features[non_noise], clusters[non_noise]),
                "n_clusters": n_clusters,
                "noise_points": sum(clusters == -1),
                "cluster_distribution": {c: sum(clusters == c) for c in set(clusters) if c != -1}
            }
    
    # Visualización mejorada
    if plot:
        plt.figure(figsize=(12, 8))
        
        # Filtrar outliers si show_noise es False
        if not show_noise:
            mask = clusters != -1
            plot_features = features[mask]
            plot_clusters = clusters[mask]
            title_noise = ""  # No mostrar conteo de ruido en título
        else:
            plot_features = features
            plot_clusters = clusters
            title_noise = f" | Noise: {sum(clusters == -1)}"
        
        # Crear scatter plot solo con clusters válidos
        scatter = plt.scatter(
            plot_features[:, 0],  # DELIQ Score
            plot_features[:, 1],  # Influence Score
            c=plot_clusters,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolor='k',
            linewidth=0.5
        )
        
        # Leyenda (solo clusters válidos)
        handles, labels = scatter.legend_elements()
        plt.legend(handles, labels, title="Clusters")
        
        plt.xlabel("DELIQ Score (scaled)")
        plt.ylabel("User Expertise (scaled)")
        plt.title(f"DBSCAN Clustering\nClusters: {n_clusters}{title_noise}")
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label="Cluster ID")
        plt.show()
    
    return {
        "clusters": clusters.tolist(),
        "dbscan_params": {"eps": eps, "min_samples": min_samples},
        **metrics
    }

def gmm_cluster_validation(results, n_components=3, plot=True, covariance_type='full', random_state=42):
    """
    Agrupa comentarios usando Gaussian Mixture Models y valida los clusters
    
    Args:
        n_components: Número de clusters a buscar
        plot: Mostrar visualización
        covariance_type: Tipo de matriz de covarianza ('full', 'tied', 'diag', 'spherical')
        random_state: Semilla para reproducibilidad
    """
    # Extraer y escalar características
    df = pd.DataFrame(results)
    # features = df[['deliq_score', 'influence_score', 'kl_divergence', 'score']].values
    features = StandardScaler().fit_transform(df[['deliq_score', 'influence_score', 'kl_divergence', 'score']])
    
    # Aplicar Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )
    clusters = gmm.fit_predict(features)
    
    # Calcular métricas
    metrics = {
        "silhouette_score": silhouette_score(features, clusters),
        "calinski_harabasz_score": calinski_harabasz_score(features, clusters),
        "n_clusters": n_components,
        "bic_score": gmm.bic(features),
        "aic_score": gmm.aic(features),
        "cluster_distribution": {c: sum(clusters == c) for c in set(clusters)}
    }
    
    # Visualización
    if plot:
        plt.figure(figsize=(12, 8))
        
        # Crear scatter plot
        scatter = plt.scatter(
            features[:, 0],  # DELIQ Score
            features[:, 1],  # Influence Score
            c=clusters,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolor='k',
            linewidth=0.5
        )
        
        # Mostrar elipses de covarianza (opcional)
        if covariance_type == 'full':
            from matplotlib.patches import Ellipse
            for i in range(n_components):
                if np.sum(clusters == i) > 0:  # Solo si el cluster tiene puntos
                    # Obtener parámetros del componente gaussiano
                    mean = gmm.means_[i, :2]
                    cov = gmm.covariances_[i][:2, :2]
                    
                    # Calcular autovalores y autovectores para la elipse
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
                    width, height = 2 * np.sqrt(eigvals)
                    
                    # Dibujar elipse
                    ell = Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        alpha=0.2,
                        color=scatter.cmap(scatter.norm(i))
                    )
                    plt.gca().add_patch(ell)
        
        # Leyenda
        handles, labels = scatter.legend_elements()
        plt.legend(handles, labels, title="Clusters")
        
        plt.xlabel("DELIQ Score (scaled)")
        plt.ylabel("Influence Score (scaled)")
        plt.title(f"GMM Clustering\nClusters: {n_components}")
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label="Cluster ID")
        plt.show()
    
    return {
        "clusters": clusters.tolist(),
        "gmm_params": {
            "n_components": n_components,
            "covariance_type": covariance_type
        },
        **metrics
    }

# NUEVAS FUNCIONES PARA OPTIMIZACIÓN DE PESOS

def evaluate_clustering_quality(weights, base_data, algorithm='dbscan', dbscan_params=None, gmm_params=None):
    """
    Evalúa la calidad del clustering para unos pesos dados
    
    Args:
        weights: Array de pesos para calcular deliq_score
        base_data: DataFrame base con los datos
        algorithm: 'dbscan' o 'gmm'
        dbscan_params: Parámetros para DBSCAN
        gmm_params: Parámetros para GMM
    
    Returns:
        Negative score (para minimización)
    """
    try:
        # Normalizar pesos para que sumen 1
        weights = weights / np.sum(weights)
        
        # Recalcular deliq_score con nuevos pesos
        df = base_data.copy()
        df['deliq_score'] = deliq_score(df[['score','kl_divergence', 'influence_score']], weights)
        
        # Convertir a array para clustering
        data_array = df.to_dict(orient='records')
        
        if algorithm == 'dbscan':
            if dbscan_params is None:
                dbscan_params = {'eps': 0.3, 'min_samples': 5}
            
            result = dbscan_cluster_validation(
                data_array, 
                eps=dbscan_params['eps'], 
                min_samples=dbscan_params['min_samples'], 
                plot=False
            )
            
            # Penalizar si hay demasiado ruido o muy pocos clusters
            n_clusters = result.get('n_clusters', 0)
            noise_ratio = result.get('noise_points', len(data_array)) / len(data_array)
            silhouette = result.get('silhouette_score', -1)
            
            if n_clusters < 2 or noise_ratio > 0.5:
                return 10  # Penalización alta
            
            # Combinar métricas (negativos porque queremos maximizar)
            score = -silhouette + noise_ratio * 2
            
        elif algorithm == 'gmm':
            if gmm_params is None:
                gmm_params = {'n_components': 3, 'covariance_type': 'full'}
            
            result = gmm_cluster_validation(
                data_array,
                n_components=gmm_params['n_components'],
                covariance_type=gmm_params['covariance_type'],
                plot=False
            )
            
            silhouette = result.get('silhouette_score', -1)
            bic = result.get('bic_score', float('inf'))
            
            # Combinar métricas (negativo para silhouette, positivo para BIC normalizado)
            score = -silhouette + (bic / len(data_array)) * 0.001
        
        return score
    
    except Exception as e:
        print(f"Error en evaluación: {e}")
        return 10  # Penalización por error

def optimize_weights_for_clustering(base_data, algorithm='both', initial_weights=None, 
                                   dbscan_params=None, gmm_params=None, n_trials=10):
    """
    Optimiza los pesos del deliq_score para mejorar el clustering
    
    Args:
        base_data: DataFrame con los datos base
        algorithm: 'dbscan', 'gmm', o 'both'
        initial_weights: Pesos iniciales (si None, usa valores aleatorios)
        dbscan_params: Parámetros para DBSCAN
        gmm_params: Parámetros para GMM
        n_trials: Número de intentos de optimización
    
    Returns:
        Diccionario con resultados de optimización
    """
    
    if initial_weights is None:
        initial_weights = np.array([0.5, 0.1, 0.4])
    
    # Configurar restricciones: pesos deben ser positivos y sumar aproximadamente 1
    bounds = [(0.01, 0.98) for _ in range(len(initial_weights))]
    constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    
    results = {}
    
    if algorithm in ['dbscan', 'both']:
        print("Optimizando pesos para DBSCAN...")
        best_score_dbscan = float('inf')
        best_weights_dbscan = initial_weights.copy()
        
        for trial in range(n_trials):
            # Usar diferentes puntos de inicio
            start_weights = np.random.dirichlet([1, 1, 1])  # Distribución uniforme en simplex
            
            try:
                result = minimize(
                    evaluate_clustering_quality,
                    start_weights,
                    args=(base_data, 'dbscan', dbscan_params, gmm_params),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraint,
                    options={'maxiter': 100}
                )
                
                if result.success and result.fun < best_score_dbscan:
                    best_score_dbscan = result.fun
                    best_weights_dbscan = result.x
                    
            except Exception as e:
                print(f"Error en trial {trial} para DBSCAN: {e}")
                continue
        
        results['dbscan'] = {
            'optimal_weights': best_weights_dbscan / np.sum(best_weights_dbscan),
            'score': best_score_dbscan
        }
    
    if algorithm in ['gmm', 'both']:
        print("Optimizando pesos para GMM...")
        best_score_gmm = float('inf')
        best_weights_gmm = initial_weights.copy()
        
        for trial in range(n_trials):
            start_weights = np.random.dirichlet([1, 1, 1])
            
            try:
                result = minimize(
                    evaluate_clustering_quality,
                    start_weights,
                    args=(base_data, 'gmm', dbscan_params, gmm_params),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraint,
                    options={'maxiter': 100}
                )
                
                if result.success and result.fun < best_score_gmm:
                    best_score_gmm = result.fun
                    best_weights_gmm = result.x
                    
            except Exception as e:
                print(f"Error en trial {trial} para GMM: {e}")
                continue
        
        results['gmm'] = {
            'optimal_weights': best_weights_gmm / np.sum(best_weights_gmm),
            'score': best_score_gmm
        }
    
    return results

def compare_clustering_results(base_data, original_weights, optimized_results, 
                             dbscan_params=None, gmm_params=None):
    """
    Compara los resultados de clustering antes y después de la optimización
    """
    print("="*60)
    print("COMPARACIÓN DE RESULTADOS DE CLUSTERING")
    print("="*60)
    
    # Parámetros por defecto
    if dbscan_params is None:
        dbscan_params = {'eps': 0.3, 'min_samples': 5}
    if gmm_params is None:
        gmm_params = {'n_components': 3, 'covariance_type': 'full'}
    
    print(f"\nPesos originales: {original_weights}")
    
    # Evaluar con pesos originales
    df_original = base_data.copy()
    df_original['deliq_score'] = deliq_score(df_original[['score','kl_divergence', 'influence_score']], original_weights)
    data_original = df_original.to_dict(orient='records')
    
    for algorithm in ['dbscan', 'gmm']:
        if algorithm in optimized_results:
            print(f"\n{algorithm.upper()} RESULTS:")
            print("-" * 30)
            
            optimal_weights = optimized_results[algorithm]['optimal_weights']
            print(f"Pesos optimizados: {optimal_weights}")
            
            # Evaluar con pesos optimizados
            df_optimized = base_data.copy()
            df_optimized['deliq_score'] = deliq_score(df_optimized[['score','kl_divergence', 'influence_score']], optimal_weights)
            data_optimized = df_optimized.to_dict(orient='records')
            
            if algorithm == 'dbscan':
                original_result = dbscan_cluster_validation(data_original, **dbscan_params, plot=False)
                optimized_result = dbscan_cluster_validation(data_optimized, **dbscan_params, plot=False)
            else:
                original_result = gmm_cluster_validation(data_original, **gmm_params, plot=False)
                optimized_result = gmm_cluster_validation(data_optimized, **gmm_params, plot=False)
            
            print(f"Silhouette Score - Original: {original_result.get('silhouette_score', 'N/A'):.4f}")
            print(f"Silhouette Score - Optimizado: {optimized_result.get('silhouette_score', 'N/A'):.4f}")
            
            if algorithm == 'dbscan':
                print(f"Noise Points - Original: {original_result.get('noise_points', 'N/A')}")
                print(f"Noise Points - Optimizado: {optimized_result.get('noise_points', 'N/A')}")
            else:
                print(f"BIC Score - Original: {original_result.get('bic_score', 'N/A'):.2f}")
                print(f"BIC Score - Optimizado: {optimized_result.get('bic_score', 'N/A'):.2f}")
            
            print(f"Calinski-Harabasz - Original: {original_result.get('calinski_harabasz_score', 'N/A'):.2f}")
            print(f"Calinski-Harabasz - Optimizado: {optimized_result.get('calinski_harabasz_score', 'N/A'):.2f}")

# EJEMPLO DE USO COMPLETO
if __name__ == "__main__":
    # Configurar parámetros para clustering
    dbscan_params = {'eps': 0.3, 'min_samples': 5}
    gmm_params = {'n_components': 3, 'covariance_type': 'full'}
    
    print("Ejecutando clustering con pesos iniciales...")
    
    # Ejecutar clustering inicial
    results_dbscan_initial = dbscan_cluster_validation(array_data, **dbscan_params, plot=True, show_noise=True)
    results_gmm_initial = gmm_cluster_validation(array_data, **gmm_params, plot=True)
    
    print("Resultados iniciales DBSCAN:", results_dbscan_initial)
    print("Resultados iniciales GMM:", results_gmm_initial)
    
    # Optimizar pesos
    print("\nIniciando optimización de pesos...")
    optimization_results = optimize_weights_for_clustering(
        merged_df, 
        algorithm='both',
        initial_weights=initial_weights,
        dbscan_params=dbscan_params,
        gmm_params=gmm_params,
        n_trials=5  # Reducido para ejemplo, aumentar para mejor optimización
    )
    
    # Mostrar resultados de optimización
    print("\nResultados de optimización:")
    for algo, results in optimization_results.items():
        print(f"{algo.upper()}: Pesos optimizados = {results['optimal_weights']}")
    
    # Comparar resultados
    compare_clustering_results(merged_df, initial_weights, optimization_results, dbscan_params, gmm_params)
    
    # Ejecutar clustering con pesos optimizados
    if 'dbscan' in optimization_results:
        print("\nEjecutando DBSCAN con pesos optimizados...")
        optimized_df_dbscan = merged_df.copy()
        optimized_df_dbscan['deliq_score'] = deliq_score(
            optimized_df_dbscan[['score','kl_divergence', 'influence_score']], 
            optimization_results['dbscan']['optimal_weights']
        )
        optimized_data_dbscan = optimized_df_dbscan.to_dict(orient='records')
        results_dbscan_optimized = dbscan_cluster_validation(optimized_data_dbscan, **dbscan_params, plot=True, show_noise=True)
    
    if 'gmm' in optimization_results:
        print("\nEjecutando GMM con pesos optimizados...")
        optimized_df_gmm = merged_df.copy()
        optimized_df_gmm['deliq_score'] = deliq_score(
            optimized_df_gmm[['score','kl_divergence', 'influence_score']], 
            optimization_results['gmm']['optimal_weights']
        )
        optimized_data_gmm = optimized_df_gmm.to_dict(orient='records')
        results_gmm_optimized = gmm_cluster_validation(optimized_data_gmm, **gmm_params, plot=True)