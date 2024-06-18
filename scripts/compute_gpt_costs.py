pip install tiktoken
pip install currencyapicom

import tiktoken
import pandas as pd
import currencyapicom

""" GPT Models Pricing - https://openai.com/api/pricing/

*   text-embedding-3-large model $0.07 (Most capable embedding model for both english and non-english tasks)
    MAX INPUT = 8191
    OUTPUT DIMENSION = 3072

*   text-embedding-3-small model $0.01 (Increased performance over text-embedding-ada-002)
    MAX INPUT = 8191
    OUTPUT DIMENSION = 1536
    
*   text-embedding-ada-002 model: $0.07 (replaced 16 first generation models)
    MAX INPUT = 8191
    OUTPUT DIMENSION = 1536

Obs: To use these models use the secondary class for GPT text embeddings
*   davinci model: $1.34
*   babbage model: $0.27 
"""

# Convert a row to text
def row_to_text(row):
    return " ".join([str(item) for item in row])

# Calculate the number of tokens
def count_tokens(text, model):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)

# Calculate cost
def calculate_cost(total_tokens, model):
    model_costs = {
        "davinci": 2.00,
        "curie": 0.006,
        "babbage": 0.40,
        "ada": 0.10,
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
    }
    cost_per_1M_tokens = model_costs[model]
    cost = (total_tokens / 1000000) * cost_per_1M_tokens
    return cost

# Get the last currency exchange rate USD to BRL
def last_exchange_rate():
    '''
    Attention!
    The base rate value corresponds to the last currency rate registereg at 23:59:59 of the day before the current date.
    '''
    client = currencyapicom.Client('cur_live_1CrTy4C1ADsdMLpFp9FkhPiWZENBjL8GXVZusBoJ')
    result = client.latest(currencies=['BRL'])
    value = float("{:.2f}".format(result['data']['BRL']['value']))
    rate_USS2BRL = value
    
    return rate_USS2BRL

# Calculate the total cost
cost = calculate_cost(total_tokens, model="babbage")
print(f"Total cost for processing the dataset with babbage model: ${cost:.2f}")


# Calculate the total cost for generating embeddings with a specified model
def compute_GPTcost (data, model):
    total_tokens = 0
    for index, row in data.iterrows():
        text = row_to_text(row)
        tokens = count_tokens(text, model)
        total_tokens += tokens
        
    cost = calculate_cost(total_tokens, model)
    print(f"Total cost for processing the dataset with %s model: US$ %.2f (R$ %.2f)." %(model,cost, cost*last_exchange_rate()))
