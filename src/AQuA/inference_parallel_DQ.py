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
        # self.weights = [0.20908452, 0.18285757]

        self.weights = [0.20908452, 0.18285757, -0.11069402, 0.29000763, 0.39535126,
        0.14655912, -0.07331445, -0.03768367, 0.07019062, -0.02847408,
        0.21126469, -0.02674237, 0.01482095, 0.00732909, -0.01900971,
        -0.04995486, -0.05884586, -0.15170863, 0.02934227, 0.10628146]

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
    "fact": "src/AQuA/trained_adapters/fact",
    "opinion": "src/AQuA/trained_adapters/opinion",
    "justification": "src/AQuA/trained_adapters/justification",
    "solproposal": "src/AQuA/trained_adapters/solproposal",
    "addknowledge": "src/AQuA/trained_adapters/addknowledge",
    "question": "src/AQuA/trained_adapters/question",
    "refusers": "src/AQuA/trained_adapters/refusers",
    "refmedium": "src/AQuA/trained_adapters/refmedium",
    "refcontents": "src/AQuA/trained_adapters/refcontents",
    "refpersonal": "src/AQuA/trained_adapters/refpersonal",
    "refformat": "src/AQuA/trained_adapters/refformat",
    "address": "src/AQuA/trained_adapters/address",
    "respect": "src/AQuA/trained_adapters/respect",
    "screaming": "src/AQuA/trained_adapters/screaming",
    "vulgar": "src/AQuA/trained_adapters/vulgar",
    "insult": "src/AQuA/trained_adapters/insult",
    "sarcasm": "src/AQuA/trained_adapters/sarcasm",
    "discrimination": "src/AQuA/trained_adapters/discrimination",

    "storytelling": "src/AQuA/trained_adapters/storytelling"
}

default_task2weights = {
    "relevance": 0.20908452,
    "fact": 0.18285757,
    "opinion": -0.11069402,
    "justification": 0.29000763,
    "solproposal":0.39535126,
    "addknowledge": 0.14655912,
    "question":-0.07331445,

    "refusers":-0.03768367,
    "refmedium":0.07019062,
    "refcontents":-0.02847408,
    "refpersonal":0.21126469,
    "refformat": -0.02674237,

    "address": 0.01482095,
    "respect": 0.00732909,
    "screaming": -0.01900971,
    "vulgar": -0.04995486,
    "insult": -0.05884586,
    "sarcasm":-0.15170863,
    "discrimination":0.02934227,

    "storytelling": 0.10628146

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