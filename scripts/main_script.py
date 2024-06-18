import torch
import torch_frame
import torch.nn as nn
import pandas as pd
from openai import OpenAI
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from typing import List
from torch import Tensor
from torch_frame import stype
from torch_frame.config import ModelConfig
from torch_frame.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

#Set your OpenAI and Voyage AI API Keys
openai_key = 'YOUR-API-KEY'
client = OpenAI(api_key=openai_key) # or simply client = OpenAI(api_key='YOUR-API-KEY')

voyageai_key = 'YOUR-API-KEY'
voyageai.api_key = voyageai_key # or simply voyageai.api_key = 'YOUR-API-KEY'

# Load your Data or Splits
train_data = pd.read_csv('train_split.csv') # change it to the actual file path
val_data = pd.read_csv('val_split.csv')
test_data = pd.read_csv('test_split.csv')

# Set your Text Embedder
em_model = 'chosen_model_name' # Example: em_model = 'rufimelo/bert-large-portuguese-cased-sts'
text_encoder = text_embedder.SentenceTransformerTextEncoder(model, device)

text_embedder_cfg = TextEmbedderConfig(text_embedder=text_encoder, batch_size=text_encoder.text_embedder_batch_size)

# Specifying Column Stypes 
col_to_stype = {"text": torch_frame.text_embedded,"toxic": torch_frame.categorical}

# Set "y" as the target column.
train_dataset = Dataset(train_data, col_to_stype=col_to_stype, target_col="toxic",split_col= None,col_to_text_embedder_cfg=text_embedder_cfg)
val_dataset = Dataset(val_data, col_to_stype=col_to_stype, target_col="toxic",split_col= None,col_to_text_embedder_cfg=text_embedder_cfg)
test_dataset = Dataset(test_data, col_to_stype=col_to_stype, target_col="toxic",split_col= None,col_to_text_embedder_cfg=text_embedder_cfg)

# Materialize each split
# Use path parameter to store generated tensor on cache dataset.materialize(path='your-path/data.pt')
train_dataset.materialize()
val_dataset.materialize()
test_dataset.materialize()

# Set up data loaders
train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame

train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

# Setting task for for classification 
is_classification = train_dataset.task_type.is_classification

output_channels = train_dataset.num_classes # {Number of different labels found in the target_col (y = toxic)}

# Setting a parser
parser = {
    'output_channels': train_dataset.num_classes,
    'col_stats': tensor_frame.col_stats,
    'col_names_dict': tensor_frame.col_names_dict,
    'em_model': 'rufimelo/bert-large-portuguese-cased-sts'
}

# Create and Compile FTT model
ftt_model = FTTransformerModel(parser)

ftt_model = torch.compile(ftt_model, dynamic=True) if args.compile else ftt_model

# Setting AdamW with Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

# Training Loop

metric = "Acc"
best_val_metric = 0
best_test_metric = 0
best_val_report = None
best_test_report = None

for epoch in range(1, args.epochs + 1):
    train_loss = ftt_model.train(epoch)
    train_results = ftt_model.test(train_loader, "Train")
    val_results = ftt_model.test(val_loader, "Validation")
    test_results = ftt_model.test(test_loader, "Test")

    if is_classification:
        train_metric = train_results["accuracy"]
        val_metric = val_results["accuracy"]
        test_metric = test_results["accuracy"]
    else:
        train_metric = train_results["rmse"]
        val_metric = val_results["rmse"]
        test_metric = test_results["rmse"]

    if is_classification and val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric
        best_val_report = val_results
        best_test_report = test_results
    elif not is_classification and val_metric < best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric
        best_val_report = val_results
        best_test_report = test_results

    print(f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
          f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}")

print(f"Best Val {metric}: {best_val_metric:.4f}, "
      f"Best Test {metric}: {best_test_metric:.4f}")


# Validation and Test Classification Reports
print("\nValidation Classification Report:")
print(classification_report(best_val_report["labels"], best_val_report["preds"], digits=4))

print("\nTest Classification Report:")
print(classification_report(best_test_report["labels"], best_test_report["preds"], digits=4))

