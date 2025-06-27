import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class DeliqClusteringAnalyzer:
    """
    Clase principal para análisis de clustering con optimización de pesos del DELIQ Score
    """

    def __init__(self, 
                 initial_weights:list[float]=None, 
                 optimal_method:str=None,
                 random_state:int=42):  # AÑADIDO: parámetro de semilla
        self.initial_weights = initial_weights
        self.random_state = random_state  # AÑADIDO
        self.expertise_order = {
            "Novice": 1, "Contributor": 2, "Expert": 3, 
            "Master": 4, "Grandmaster": 5, None: 0
        }
        self.data = None
        self.optimal_method = optimal_method
        
        # AÑADIDO: Configurar semillas aleatorias
        np.random.seed(self.random_state)
        
    def load_and_merge_features(self, 
                                dq_path='src/results/processed_data_inferenced.csv',
                                kl_path='src/results/kl_content_vs_domain.xlsx',
                                pr_path='src/results/ranked_users.xlsx', 
                                outliers_field='votes'):
        """Carga y combina las características DQ, KL, PR"""
        # Load DQ data
        df_dq = pd.read_csv(dq_path, sep='\t')
        df_dq = df_dq.drop(columns=['content', 'tags', 'datetime'], errors='ignore')
        
        # Load KL data
        df_kl = pd.read_excel(kl_path)
        df_kl = df_kl.loc[df_kl.groupby('user')['kl_divergence'].idxmin()].sort_values(by='id')
        
        # Load PR data
        df_pr = pd.read_excel(pr_path)
        df_pr = df_pr[['username', 'influence_score', 'expertise', 'received_post_votes', 'rank']]
        df_pr.rename(columns={'username': 'user'}, inplace=True)
        
        # Merge all datasets
        merged_df = pd.merge(df_dq, df_kl[['id', 'kl_divergence']], on='id', how='left')
        merged_df = pd.merge(merged_df, df_pr[['user', 'influence_score']], on='user', how='left')
        merged_df[['kl_divergence', 'influence_score']] = merged_df[['kl_divergence', 'influence_score']].fillna(0)
        
        # Add expertise ordinal
        merged_df['expertise_ordinal'] = merged_df['expertise'].map(self.expertise_order).fillna(0)
        
        # Identificar outliers por método intercuartílico (IQR)
        q1 = merged_df[outliers_field].quantile(0.25)
        q3 = merged_df[outliers_field].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        merged_df['outlier'] = ~merged_df[outliers_field].between(lower_bound, upper_bound)

        merged_df = merged_df[merged_df['outlier'] == False]  # Exclude outliers
        self.data = merged_df
        return merged_df
    
    def calculate_deliq_score(self, weights=None):
        """Calcula el DELIQ score usando los pesos especificados"""
        if weights is None:
            weights = self.initial_weights

        if self.data is None:
            raise ValueError("Datos no cargados. Ejecuta load_and_merge_features() primero.")

        # Comprobar que los pesos estén normalizados
        weights = weights / np.sum(weights)

        # Calcular DELIQ score
        features = self.data[['score', 'kl_divergence', 'influence_score']].values
        self.data['deliq_score'] = np.dot(features, weights)

        return self.data
    
    def prepare_clustering_data(self, 
                                features_list=['deliq_score', 'n_comments', 'votes'], 
                                standardize=False):
        """Prepara los datos para clustering"""
        if self.data is None:
            raise ValueError("Datos no cargados.")
            
        # Verificar que las columnas existen
        missing_cols = [col for col in features_list if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
            
        features = self.data[features_list].values
        
        if standardize:
            # MODIFICADO: Usar semilla para StandardScaler si es posible
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            
        return features, features_list
    
    def dbscan_clustering(self, 
                          features_list=['deliq_score', 'n_comments', 'votes'],
                          eps=0.7, 
                          min_samples=10, 
                          standardize=False):
        """Ejecuta clustering DBSCAN"""
        features, feature_names = self.prepare_clustering_data(features_list, standardize)
        
        # Aplicar DBSCAN
        dbscan = DBSCAN(eps=eps, 
                        min_samples=min_samples)
        clusters = dbscan.fit_predict(features)
        
        # Calcular métricas
        metrics = self._calculate_clustering_metrics(features, clusters, algorithm='dbscan')
        
        return {
            'clusters': clusters,
            'features': features,
            'feature_names': feature_names,
            'params': {'eps': eps, 'min_samples': min_samples},
            **metrics
        }
    
    def gmm_clustering(self, features_list=['deliq_score', 'n_comments', 'votes'],
                      n_components=3, covariance_type='full', standardize=False, 
                      random_state=None):  # MODIFICADO: usar random_state de la clase
        """Ejecuta clustering GMM"""
        if random_state is None:
            random_state = self.random_state  # AÑADIDO
            
        features, feature_names = self.prepare_clustering_data(features_list, standardize)
        
        # Aplicar GMM
        gmm = GaussianMixture(n_components=n_components, 
                             covariance_type=covariance_type,
                             random_state=random_state)
        clusters = gmm.fit_predict(features)
        
        # Calcular métricas
        metrics = self._calculate_clustering_metrics(features, clusters, algorithm='gmm')
        metrics.update({
            'bic_score': gmm.bic(features),
            'aic_score': gmm.aic(features)
        })
        
        return {
            'clusters': clusters,
            'features': features,
            'feature_names': feature_names,
            'params': {'n_components': n_components, 'covariance_type': covariance_type},
            'gmm_model': gmm,
            **metrics
        }
    
    def _calculate_clustering_metrics(self, 
                                      features, 
                                      clusters, 
                                      algorithm='dbscan'):
        """Calcula métricas de calidad del clustering"""
        metrics = {}
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        if algorithm == 'dbscan':
            noise_points = sum(clusters == -1)
            metrics['noise_points'] = noise_points
            metrics['noise_ratio'] = noise_points / len(clusters)
            
            # Solo calcular métricas si hay clusters válidos
            if n_clusters > 1:
                non_noise = clusters != -1
                if sum(non_noise) > 1:
                    metrics.update({
                        'silhouette_score': silhouette_score(features[non_noise], clusters[non_noise]),
                        'calinski_harabasz_score': calinski_harabasz_score(features[non_noise], clusters[non_noise])
                    })
            else:
                metrics['silhouette_score'] = -1
                metrics['calinski_harabasz_score'] = -1
        else:  # GMM
            if n_clusters > 1:
                metrics.update({
                    'silhouette_score': silhouette_score(features, clusters),
                    'calinski_harabasz_score': calinski_harabasz_score(features, clusters)
                })
        
        metrics['n_clusters'] = n_clusters
        metrics['cluster_distribution'] = {c: sum(clusters == c) for c in set(clusters) if c != -1}
        
        return metrics
    
    def plot_clustering_results(self, result, plot_features=[0, 2], show_noise=False, 
                               figsize=(12, 8)):
        """Visualiza los resultados del clustering"""
        clusters = result['clusters']
        features = result['features']
        feature_names = result['feature_names']
        
        plt.figure(figsize=figsize)
        
        # Filtrar ruido si es necesario
        if not show_noise and -1 in clusters:
            mask = clusters != -1
            plot_features_data = features[mask]
            plot_clusters = clusters[mask]
            title_noise = ""
        else:
            plot_features_data = features
            plot_clusters = clusters
            title_noise = f" | Noise: {result.get('noise_points', 0)}" if 'noise_points' in result else ""
        
        # Crear scatter plot
        scatter = plt.scatter(
            plot_features_data[:, plot_features[0]],
            plot_features_data[:, plot_features[1]],
            c=plot_clusters,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolor='k',
            linewidth=0.5
        )
        
        # Configurar plot
        plt.xlabel(feature_names[plot_features[0]])
        plt.ylabel(feature_names[plot_features[1]])
        plt.title(f"Clustering Results\nClusters: {result['n_clusters']}{title_noise}")
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label="Cluster ID")
        
        # Leyenda
        handles, labels = scatter.legend_elements()
        plt.legend(handles, labels, title="Clusters")
        
        plt.tight_layout()
        plt.show()
    
    def _objective_function(self, weights, algorithm_params):
        """Función objetivo para optimización de pesos"""
        try:
            # Recalcular DELIQ score con nuevos pesos
            self.calculate_deliq_score(weights)
            
            algorithm = algorithm_params['algorithm']
            params = algorithm_params['params']
            
            # Ejecutar clustering
            if algorithm == 'dbscan':
                result = self.dbscan_clustering(**params)
                
                # Penalizar configuraciones problemáticas
                if result['n_clusters'] < 2 or result.get('noise_ratio', 1) > 0.5:
                    return 10
                
                # Objetivo: maximizar silhouette, minimizar ruido
                silhouette = result.get('silhouette_score', -1)
                noise_ratio = result.get('noise_ratio', 1)
                score = -silhouette + noise_ratio * 2
                # score = noise_ratio
                
            else:  # GMM
                result = self.gmm_clustering(**params)
                
                if result['n_clusters'] < 2:
                    return 10
                
                # Objetivo: maximizar silhouette, minimizar BIC
                silhouette = result.get('silhouette_score', -1)
                bic = result.get('bic_score', float('inf'))
                score = -silhouette + (bic / len(self.data)) * 0.001
                # score = bic
            
            return score
            
        except Exception as e:
            print(f"Error en función objetivo: {e}")
            return 10
    
    def optimize_weights(self, 
                         algorithm='both', 
                         n_trials=10, 
                        dbscan_params=None, 
                        gmm_params=None):
        """Optimiza los pesos del DELIQ score"""
        
        # AÑADIDO: Configurar semilla para esta optimización específica
        np.random.seed(self.random_state)
        
        # Parámetros por defecto
        dbscan_params = dbscan_params or {'eps': 0.7, 'min_samples': 10}
        gmm_params = gmm_params or {'n_components': 3, 'covariance_type': 'diag'}
        
        # Configurar optimización
        bounds = [(0.01, 0.98) for _ in range(len(self.initial_weights))]
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            # {'type': 'ineq', 'fun': lambda x: 0.25 - x[1]}  # KL divergence <= 25%
        ]
        
        results = {}
        
        algorithms_to_run = [algorithm] if algorithm in ['dbscan', 'gmm'] else ['dbscan', 'gmm']
        
        for algo in algorithms_to_run:
            print(f"Optimizando pesos para {algo.upper()}...")
            
            best_score = float('inf')
            best_weights = self.initial_weights.copy()
            
            params = dbscan_params if algo == 'dbscan' else gmm_params
            algorithm_params = {'algorithm': algo, 'params': params}
            
            for trial in range(n_trials):
                # MODIFICADO: Punto de inicio determinístico basado en trial
                np.random.seed(self.random_state + trial)  # Semilla única por trial
                start_weights = np.random.dirichlet([1, 1, 1])
                
                try:
                    result = minimize(
                        self._objective_function,
                        start_weights,
                        args=(algorithm_params,),
                        method=self.optimal_method,
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 500}
                    )
                    
                    if result.success and result.fun < best_score:
                        best_score = result.fun
                        best_weights = result.x
                        iters = result.nit
                        
                except Exception as e:
                    print(f"Error en trial {trial} para {algo}: {e}")
                    continue
            
            results[algo] = {
                'optimal_weights': best_weights / np.sum(best_weights),
                'score': best_score, 
                'n_iterations': iters
            }
        
        return results
    
    def compare_results(self, 
                        original_weights,
                        optimal_method, 
                        optimized_results, 
                        dbscan_params=None, 
                        gmm_params=None):
        """Compara resultados antes y después de la optimización"""
        optimal_method = self.optimal_method if optimal_method is None else optimal_method
        dbscan_params = dbscan_params or {'eps': 0.7, 'min_samples': 10}
        gmm_params = gmm_params or {'n_components': 3, 'covariance_type': 'full'}
        
        print("="*60)
        print("COMPARACIÓN DE RESULTADOS DE CLUSTERING")
        print("="*60)
        print(f"Pesos originales: {original_weights}")
        print(f"Método: {optimal_method}")
        print(f"Random State: {self.random_state}")  # AÑADIDO
        
        for algorithm in ['dbscan', 'gmm']:
            if algorithm not in optimized_results:
                continue
                
            print(f"\n{algorithm.upper()} RESULTS:")
            print("-" * 30)
            
            optimal_weights = optimized_results[algorithm]['optimal_weights']
            print(f"Pesos optimizados: {optimal_weights}")
            
            # Evaluar con pesos originales
            self.calculate_deliq_score(original_weights)
            if algorithm == 'dbscan':
                original_result = self.dbscan_clustering(**dbscan_params)
            else:
                original_result = self.gmm_clustering(**gmm_params)
            
            # Evaluar con pesos optimizados
            self.calculate_deliq_score(optimal_weights)
            if algorithm == 'dbscan':
                optimized_result = self.dbscan_clustering(**dbscan_params)
            else:
                optimized_result = self.gmm_clustering(**gmm_params)
            
            # Mostrar comparación
            self._print_comparison_metrics(original_result, optimized_result, algorithm)
    
    def _print_comparison_metrics(self, original, optimized, algorithm):
        """Imprime métricas de comparación"""
        metrics_to_compare = ['silhouette_score', 'calinski_harabasz_score']
        
        if algorithm == 'dbscan':
            metrics_to_compare.extend(['noise_points', 'noise_ratio'])
        else:
            metrics_to_compare.extend(['bic_score', 'aic_score'])
        
        for metric in metrics_to_compare:
            orig_val = original.get(metric, 'N/A')
            opt_val = optimized.get(metric, 'N/A')
            
            if isinstance(orig_val, (int, float)) and isinstance(opt_val, (int, float)):
                print(f"{metric} - Original: {orig_val:.4f}, Optimizado: {opt_val:.4f}")
            else:
                print(f"{metric} - Original: {orig_val}, Optimizado: {opt_val}")

# USO MODIFICADO
if __name__ == "__main__":
    # AÑADIDO: Configurar semilla global
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    
    # Inicializar analizador
    methods = ['SLSQP', 'L-BFGS-B', 'CG']
    # methods = ['SLSQP', 'L-BFGS-B']
    outliers_field = 'votes'  # Cambiar según el campo de outliers deseado
    comparison_table = []

    for method in methods:
        print(f"\n{'='*20} OPTIMIZATION METHOD: {method} {'='*20}")
        # MODIFICADO: Añadir random_state
        analyzer = DeliqClusteringAnalyzer(initial_weights=[0.35, 0.35, 0.30],
                                           optimal_method=method,
                                           random_state=RANDOM_STATE)
        
        # Cargar datos
        data = analyzer.load_and_merge_features(outliers_field=outliers_field)
        
        # Calcular DELIQ score inicial
        analyzer.calculate_deliq_score()
        
        # Parámetros de clustering
        dbscan_params = {'eps': 0.7, 'min_samples': 10, 'standardize': False}
        gmm_params = {'n_components': 4, 'covariance_type': 'full', 'standardize': False, 'random_state': RANDOM_STATE}
        
        # Clustering inicial
        dbscan_initial = analyzer.dbscan_clustering(**dbscan_params)
        gmm_initial = analyzer.gmm_clustering(**gmm_params)
        
        # Optimizar pesos
        optimization_results = analyzer.optimize_weights(
            algorithm='both',
            n_trials=20,
            dbscan_params=dbscan_params,
            gmm_params=gmm_params
        )
        
        # Evaluar con pesos optimizados
        for algo, initial_result in zip(['dbscan', 'gmm'], [dbscan_initial, gmm_initial]):
            if algo in optimization_results:
                optimal_weights = optimization_results[algo]['optimal_weights']
                optimal_iters = optimization_results[algo]['n_iterations']
                analyzer.calculate_deliq_score(optimal_weights)
                if algo == 'dbscan':
                    final_result = analyzer.dbscan_clustering(**dbscan_params)
                else:
                    final_result = analyzer.gmm_clustering(**gmm_params)
                # Calculate improvement percentage
                init_sil = initial_result.get('silhouette_score', float('nan'))
                final_sil = final_result.get('silhouette_score', float('nan'))
                if np.isnan(init_sil) or init_sil == 0:
                    improvement_pct = float('nan')
                else:
                    improvement_pct = 100 * (final_sil - init_sil) / abs(init_sil)
        
                comparison_table.append({
                    'Method': method,
                    'Algorithm': algo.upper(),
                    'Outlier Field': outliers_field,
                    'Initial Silhouette': init_sil,
                    'Final Silhouette': final_sil,
                    'Improvement (%)': improvement_pct,
                    'Optimized Weights': np.round(optimal_weights, 4),
                    'Max Iter Achieved': optimal_iters,
                    'Initial Result': initial_result,
                    'Final Result': final_result,
                    'Analyzer': analyzer  # Save analyzer for plotting
                })

    # Imprimir tabla comparativa
    print("\n" + "="*80)
    print("COMPARATIVE TABLE OF OPTIMIZATION METHODS")
    print("="*80)
    print(f"Random State Used: {RANDOM_STATE}")  # AÑADIDO
    print(f"{'Method':<10} {'Algorithm':<8} {'Outlier Field':<14} {'Init Silh.':<12} {'Final Silh.':<12} {'% Improv.':<12} {'Max Iter':<10} {'Optimized Weights'}")
    for row in comparison_table:
        init_sil = row['Initial Silhouette']
        final_sil = row['Final Silhouette']
        improvement = row['Improvement (%)']
        maxiter = row['Max Iter Achieved']
        print(f"{row['Method']:<10} {row['Algorithm']:<8} {row['Outlier Field']:<14} "
              f"{init_sil:<12.4f} {final_sil:<12.4f} "
              f"{improvement if not np.isnan(improvement) else 'N/A':<12.2f} "
              f"{maxiter:<10} {row['Optimized Weights']}")

    # Detectar el mejor % Improvement
    best_row = max(
        (r for r in comparison_table if not np.isnan(r['Improvement (%)'])),
        key=lambda r: r['Improvement (%)'],
        default=0
    )

    if best_row:
        print("\n" + "="*40)
        print("BEST IMPROVEMENT FOUND:")
        print("="*40)
        print(f"Method: {best_row['Method']}, Algorithm: {best_row['Algorithm']}, Improvement: {best_row['Improvement (%)']:.2f}%")
        analyzer = best_row['Analyzer']
        print("Plotting clustering results before optimization...")
        analyzer.plot_clustering_results(best_row['Initial Result'])
        print("Plotting clustering results after optimization...")
        analyzer.plot_clustering_results(best_row['Final Result'])
    else:
        print("No valid improvement found.")

    # Guardar datos
    data.to_excel('src/results/data_deliq_scores.xlsx', index=False)