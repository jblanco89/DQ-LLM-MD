from transformers import AutoTokenizer, AutoConfig, AutoAdapterModel, AdapterTrainer
from AQuA.data import InferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
import os
from AQuA.utils import get_dynamic_parallel

class AQuAPredictor:
    def __init__(self, model_name="google-bert/bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name).to(self.device)
        
        # Valores por defecto (pueden ser sobrescritos)
        self.weights = [0.20908452, 0.18285757]
        self.maxval = 4.989267539999999
        self.minval = -1.66928295
        self.minmaxdif = self.maxval - self.minval
        
        self.root_dir = 'src/AQuA/'

    def load_adapters(self, task2identifier):
        """Carga todos los adaptadores especificados"""
        self.task2identifier = task2identifier
        adapter_counter = 0
        
        for k, v in self.task2identifier.items():
            print(f"Loading adapter {k} as adapter {adapter_counter}")
            # Asegurar que las rutas no tengan espacios
            adapter_path = v.replace(" ", "_")
            try:
                self.model.load_adapter(
                    adapter_path,
                    load_as=f"adapter{adapter_counter}",
                    with_head=True,
                    set_active=True,
                    source="hf"
                )
                adapter_counter += 1
            except Exception as e:
                print(f"Error loading adapter {k}: {str(e)}")
                continue
        
        print(f"Loaded {adapter_counter} adapters")
        self.model.active_adapters = get_dynamic_parallel(adapter_counter)
        return self

    def normalize_scores(self, scores, bound=5):
        """Normaliza los scores entre 0 y el límite especificado"""
        return ((scores - self.minval) / self.minmaxdif) * bound

    def predict(self, dataset_path, text_col, batch_size=8, output_path=None):
        """Ejecuta la predicción completa"""
        # Preparar el dataset
        dataset = InferenceDataset(
            path_to_dataset=dataset_path,
            tokenizer=self.tokenizer,
            text_col=text_col
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Diccionario para almacenar resultados
        output_dic = {k: [] for k in self.task2identifier.keys()}
        
        # Procesamiento por lotes
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Mover datos al dispositivo
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
            
             # MODIFICACIÓN PRINCIPAL AQUÍ:
            if isinstance(outputs, torch.Tensor):
                # Si outputs es un tensor directo
                for i, task in enumerate(self.task2identifier.keys()):
                    predictions = outputs  # Usamos el tensor directamente
                    prediction = torch.argmax(predictions, dim=1).detach().cpu().numpy()
                    output_dic[task].extend(prediction)
            else:
                # Si outputs es una secuencia o objeto con logits
                for i, task in enumerate(self.task2identifier.keys()):
                    if hasattr(outputs[i], 'logits'):
                        predictions = outputs[i].logits
                    else:
                        predictions = outputs[i]
                    prediction = torch.argmax(predictions, dim=1).detach().cpu().numpy()
                    output_dic[task].extend(prediction)

        
        # Post-procesamiento
        for task, preds in output_dic.items():
            dataset.dataset[task + "_ad"] = preds
        
        # Calcular score ponderado
        adapters = self.task2identifier.keys()
        score = dataset.dataset[[s + "_ad" for s in adapters]].dot(self.weights)
        dataset.dataset["score"] = self.normalize_scores(score)
        
        # Guardar resultados si se especifica path
        if output_path:
            dataset.dataset.to_csv(output_path, sep="\t", index=False)
        
        return dataset.dataset

# Configuración por defecto (para compatibilidad con el script original)
default_task2identifier = {
    "relevance": "src/AQuA/trained_adapters/relevance",
    "fact": "src/AQuA/trained_adapters/fact"
}

default_task2weights = {
    "relevance": 0.20908452,
    "fact": 0.18285757
}

def run_as_script():
    """Función para mantener compatibilidad con ejecución como script"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inference_data', type=str, help='path to the test data')
    parser.add_argument('text_col', type=str, help="column name of text column")
    parser.add_argument('batch_size', type=int)
    parser.add_argument("output_path", type=str, help="path to output file")
    args = parser.parse_args()

    predictor = AQuAPredictor()
    predictor.load_adapters(default_task2identifier)
    predictor.predict(
        dataset_path=args.inference_data,
        text_col=args.text_col,
        batch_size=args.batch_size,
        output_path=args.output_path
    )

if __name__ == '__main__':
    run_as_script()