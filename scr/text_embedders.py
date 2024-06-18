import torch
import voyageai
import pandas as pd
from typing import List
from torch import Tensor
from openai import OpenAI
from openai import Embedding
from voyageai import get_embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BertModel, BertTokenizer, DebertaTokenizer, DebertaV2Model, DebertaV2ForMaskedLM

#BaseTextEncoder provides common functionality like tokenization and encoding sentences.
class BaseTextEncoder:
    def __init__(self, model_name: str, device: torch.device):
        self.tokenizer = self.load_tokenizer(model_name)
        self.model = self.load_model(model_name).to(device)
        self.device = device

    def load_tokenizer(self, model_name: str):
        raise NotImplementedError

    def load_model(self, model_name: str):
        raise NotImplementedError

    def  __call__(self, sentences: List[str]) -> Tensor:
        raise NotImplementedError

    def extract_embeddings(self, model_output, attention_mask: Tensor) -> Tensor:
        raise NotImplementedError

# Encoder for BERT-based models like BERTimbau,        
class BertTextEncoder(BaseTextEncoder):
    def load_tokenizer(self, model_name: str):
        return AutoTokenizer.from_pretrained(model_name)

    def load_model(self, model_name: str):
        return BertModel.from_pretrained(model_name)

    def __call__(self, sentences: List[str]) -> Tensor:
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.extract_embeddings(outputs, inputs["attention_mask"])

    def extract_embeddings(self, model_output, attention_mask: Tensor) -> Tensor:
        cls_embeddings = model_output.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1).expand(cls_embeddings.size())
        sum_embeddings = torch.sum(cls_embeddings * attention_mask, dim=1)
        mean_embeddings = sum_embeddings / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        return mean_embeddings


# Encoder for BERT-DeBERTa models like Albertina PT* & DeBERTa-V2-XL
# Obs: Avg. Hidden States embeddings
class AlbertinaTextEncoder(BaseTextEncoder):
    # Model Handler for AlbertinaPT* and DeBERTa models
    def modelHandler:(self):
        self.model_tokenizer_mapping = {
            'PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac': 'PORTULAN/albertina-ptbr-nobrwac',
            'PORTULAN/albertina-900m-portuguese-ptbr-encoder': 'microsoft/deberta-v2-xlarge',
            'PORTULAN/albertina-ptbr-nobrwac': 'PORTULAN/albertina-ptbr-nobrwac',
            'microsoft/deberta-xlarge-mnli': 'microsoft/deberta-xlarge-mnli',
            # Add other model-tokenizer pairs as needed
        }
        
    def load_model(self, model_name: str):
        return DebertaV2Model.from_pretrained(model_name).to(device)

    def load_tokenizer(self, model_name: str):
        # Get the tokenizer name from the dictionary
        tokenizer_name = self.model_tokenizer_mapping.get(model_name)
        if tokenizer_name is None:
            raise ValueError(f"No tokenizer found for model {model_name}")
        
        # Load and return the tokenizer
        return AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, sentences: List[str]) -> Tensor:
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.extract_embeddings(outputs, inputs["attention_mask"]).cpu()

    def extract_embeddings(self, model_output, attention_mask: Tensor) -> Tensor:
        last_hidden_state = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        mean_embeddings = sum_embeddings / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return mean_embeddings

class SentenceTransformerTextEncoder(BaseTextEncoder):
    def __init__(self, model_name: str, device: torch.device):
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, sentences: List[str]) -> Tensor:
        # Encode a list of batch_size sentences into a PyTorch Tensor of size [batch_size, emb_dim]
        # output_value parameter can be “sentence_embedding” (default), "token_embeddings" or None
        embeddings = self.model.encode(sentences, convert_to_numpy=False, convert_to_tensor=True) 
        return embeddings.cpu()

class GPTTextEncoder(BaseTextEncoder):
    dimension: int = 1536
    text_embedder_batch_size: int = 25
    
    def __init__(self, model_name: str):
        self.model = model_name

    def __call__(self, sentences: list) -> torch.Tensor:
        items: List[Embedding] = client.embeddings.create(
            input=sentences, model=self.model).data
        assert len(items) == len(sentences)
        embeddings = [torch.FloatTensor(item.embedding).view(1, -1) for item in items]

        return torch.cat(embeddings, dim=0)
        
class VoyageAIEmbedding(BaseTextEncoder):
    dimension: int = 1024
    text_embedder_batch_size: int = 20

    def __init__(self, model: str, api_key: str):
        self.model = model

    def __call__(self, sentences: List[str]) -> Tensor:
        voyageai.api_key = api_key

        items: List[List[float]] = get_embeddings(sentences, model=self.model)
        assert len(items) == len(sentences)
        embeddings = [torch.FloatTensor(item).view(1, -1) for item in items]
        return torch.cat(embeddings, dim=0)