import pandas as pd
from AQuA.inference_parallel_DQ import AQuAPredictor
import pandas as pd

root_dir = 'src/AQuA/'

def convert_excel_to_tsv(input_file, output_file):
    df = pd.read_excel(input_file)
    df.to_csv(output_file, sep='\t', index=False)
    
if __name__ == '__main__':
    input_file_xlsx = 'src/results/processed_data.xlsx'
    input_file_csv = 'src/results/processed_data.csv'
    convert_excel_to_tsv(input_file_xlsx, input_file_csv)

    # Configuración personalizada
    custom_task2identifier = {
        "relevance": root_dir+ "trained_adapters/relevance",
        "fact": root_dir+ "trained_adapters/fact",
        # "opinion": root_dir + "trained_adapters/opinion",
        # "justification": root_dir + "trained_adapters/justification",
        # "solproposal": root_dir + "trained_adapters/solproposal",
        # "addknowledge": root_dir + "trained_adapters/addknowledge",
        # "question": root_dir + "trained_adapters/question"
    }

    # 1. Inicialización
    predictor = AQuAPredictor()

    # 2. Configurar parámetros (opcional)
    predictor.weights = [0.20908452, 0.18285757] 
    # -0.11069402, 0.29000763, 0.39535126,
    #     0.14655912, -0.07331445] 
    predictor.minval = -1.66928295                # Ajustar parámetros de normalización
    predictor.maxval = 4.989267539999999

    # 3. Cargar adaptadores
    predictor.load_adapters(custom_task2identifier)

    # 4. Ejecutar inferencia
    results = predictor.predict(
        dataset_path=input_file_csv,
        text_col="content",
        batch_size=8,
        output_path="src/results/processed_data_inferenced.csv"  # Opcional
    )

    # 5. Trabajar con los resultados
    print("Primeras filas de resultados:")
    print(results.head())

   
    

