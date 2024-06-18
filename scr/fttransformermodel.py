import torch
import torch_frame
import torch.nn as nn

from tqdm import tqdm
from typing import List
from torch import Tensor
from torch_frame import stype
from torch_frame.config import ModelConfig
from torch_frame.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from torch_frame.nn import (LinearEncoder,EmbeddingEncoder,LinearEmbeddingEncoder,FTTransformer)

class FTTransformerModel:
    def __init__(self, parser):
        self.channels = 256
        self.num_layers = 12
        self.batch_size = 512
        self.lr = 0.0001
        self.epochs = 100
        self.seed = 0

        # Retrieve parameters from the parser
        self.out_channels = parser.get('out_channels')
        self.col_stats = parser.get('col_stats')
        self.col_names_dict = parser.get('col_names_dict')
        
        em_model = parser.get('em_model')
        self.stype_encoder_dict = self.textEmbedderHandler(em_model)
        
        # Create the model
        self.model = self.create_model()
        
    def create_model(self):
        return FTTransformer(
            channels=self.channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            col_stats=self.col_stats,
            col_names_dict=self.col_names_dict,
            stype_encoder_dict=self.stype_encoder_dict
        )

        
    def textEmbedderHandler(self, em_model):
        # Define the embedding model to output dimension mapping
        embmodel_outdim_mapping = {
            'text-embedding-3-large': 1536,
            'voyage-large-2': 1536,
            'PORTULAN/albertina-900m-portuguese-ptbr-encoder': 1536,
            'neuralmind/bert-large-portuguese-cased': 1024,
            'microsoft/deberta-xlarge-mnli': 1536,
            'intfloat/multilingual-e5-large': 1024,
            'rufimelo/bert-large-portuguese-cased-sts': 1024,
            # Add other embedding model-out-dim pairs as needed
        }
        
        out_channels = embmodel_outdim_mapping.get(em_model, 1024)  # Default to 1024 if model not found
        
        return {
            'categorical': EmbeddingEncoder(),
            'embedding': LinearEmbeddingEncoder(out_channels=out_channels)
        }
        
    def train(epoch: int) -> float:
        model.train()
        loss_accum = total_count = 0
        all_preds = []
        all_labels = []
    
        for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
            tf = tf.to(device)
            pred = model(tf)
            if is_classification:
                loss = F.cross_entropy(pred, tf.y)
                all_preds.extend(pred.argmax(dim=-1).cpu().numpy())
            else:
                loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(tf.y)
            total_count += len(tf.y)
            optimizer.step()
            scheduler.step()
            all_labels.extend(tf.y.cpu().numpy())
    
        return loss_accum / total_count

    def test(loader: DataLoader, split_name: str) -> dict:
        model.eval()
        accum = total_count = 0
        all_preds = []
        all_labels = []
    
        for tf in loader:
            tf = tf.to(device)
            pred = model(tf)
            if is_classification:
                pred_class = pred.argmax(dim=-1)
                all_preds.extend(pred_class.cpu().numpy())
                all_labels.extend(tf.y.cpu().numpy())
                accum += float((tf.y == pred_class).sum())
            else:
                accum += float(
                    F.mse_loss(pred.view(-1), tf.y.view(-1), reduction="sum"))
            total_count += len(tf.y)
    
        if is_classification:
            accuracy = accum / total_count
            f1 = f1_score(all_labels, all_preds, average='weighted')
            return {"accuracy": accuracy, "f1": f1, "preds": all_preds, "labels": all_labels}
        else:
            rmse = (accum / total_count)**0.5
            return {"rmse": rmse}    
