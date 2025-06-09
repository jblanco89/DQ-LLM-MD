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
        "opinion": root_dir + "trained_adapters/opinion",
        "justification": root_dir + "trained_adapters/justification",
        "solproposal": root_dir + "trained_adapters/solproposal",
        "addknowledge": root_dir + "trained_adapters/addknowledge",
        "question": root_dir + "trained_adapters/question",
        "refusers": root_dir + "trained_adapters/refusers",
        "refmedium": root_dir + "trained_adapters/refmedium",
        "refcontents": root_dir + "trained_adapters/refcontents",
        "refpersonal": root_dir + "trained_adapters/refpersonal",
        "refformat": root_dir + "trained_adapters/refformat",
        "address": root_dir + "trained_adapters/address",
        "respect": root_dir + "trained_adapters/respect",
        "screaming": root_dir + "trained_adapters/screaming",
        "vulgar": root_dir + "trained_adapters/vulgar",
        "insult": root_dir + "trained_adapters/insult",
        "sarcasm": root_dir + "trained_adapters/sarcasm",
        "discrimination": root_dir + "trained_adapters/discrimination",
        "storytelling": root_dir + "trained_adapters/storytelling"

    }

    # 1. Inicialización
    predictor = AQuAPredictor()

    # 2. Configurar parámetros (opcional)
    # predictor.weights = [0.20908452, 0.18285757]
    predictor.weights = [0.20908452, 0.18285757, -0.11069402, 0.29000763, 0.39535126,
        0.14655912, -0.07331445, -0.03768367, 0.07019062, -0.02847408,
        0.21126469, -0.02674237, 0.01482095, 0.00732909, -0.01900971,
        -0.04995486, -0.05884586, -0.15170863, 0.02934227, 0.10628146]

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

   
    

