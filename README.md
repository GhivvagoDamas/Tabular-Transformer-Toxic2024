# A Transformer-based Tabular Approach to Detect Toxic Comments

---
## Abstract
<p align="justify">  </p>

<!--- <p align="justify"> This repository contains the code and resources for detecting toxic and hateful speech on social media, focusing 
on Brazilian Portuguese comments. The approach utilizes the Tabular Deep Learning model FT-Transformer and various embedding models 
to enhance the detection accuracy of toxic content. </p> --->
---
## Requeriments

Make sure to use Python 3.10.4.

This work used Google Colab as an environment, mainly using the T4 free GPU tier.
Using a decent GPU is heavily encouraged.

Install required Python dependencies.

```bash
!pip install -r requirements.txt
```
 Optional python packages:

Dataset

ToLD-Br dataset is available at [here](https://github.com/JAugusto97/ToLD-Br) and on [HuggingFace](https://huggingface.co/datasets/JAugusto97/told-br). 
However, you can find the original ToLD-Br dataset files, splits, and our binary two column (text,label) version [here](data/).


<!--- Download and move to your current directory utils.py file --->

To run the experiment use the Python Notebook available xxxxxx.


An API Key is required to reproduce results with **OpenAI** and **VoyageAI** Embedding Models (EM).

For the VoyageAI *voyage-large-2* embedding model processing the first 50MM tokens is free (including other EM);
In the case of OpenAI, some credits need to be altho
check the usage cost of the model or refer to xxxxx to calculate the cost based on your dataset size.
pip install tiktoken



## List of Embedding Models
Source
| Alias   | Model Name | Language Support | Model Type | Output Dim |
| :---          |     :---:      |     :---:     |     :---:      |          ---: |
| BERTimbau     | neuralmind/bert-large-portuguese-cased     | Monolingual | BERT     | 1024    |
| AlbertinaPTBR | PORTULAN/albertina-900m-portuguese-ptbr-encoder       | Monolingual | BERT/DeBERTaV2      | 1536      |
| SBERTimbau    | rufimelo/bert-large-portuguese-cased-sts       | Monolingual | SBERT       | 1024      |
| ME5Large    | intfloat/multilingual-e5-large       | Multilingual | SBERT       | 1024      |
| DeBERTaV2XL     | microsoft/deberta-v2-xlarge-mnli       | Multilingual | BERT/DeBERTaV2       | 1536      |
| VoyageLarge2     | voyage-large-2      | Multilingual | LLM Emb       | 1536      |
| OpenAI-TE3-large     | text-embedding-3-large | Multilingual | LLM Emb       | 1536      |

## Running the experiment

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```
## Results

Test Acc	Method	Model Name	Source
| Text Embedder | Test F1 (macro avg) | Test Acc | Precision | Recall |
| :---          |     :---:      |     :---:     |     :---:      |          ---: |
| BERTimbau     | 0.7316 | 0.7333 | BERT     | 1024    |
| AlbertinaPTBR    | 0,7066 | 0.6906 | BERT/DeBERTaV2      | 1536      |
| SBERTimbau    | 0.7317 | 0.7319 | LLM Emb       | 1536      |
| ME5Large    | 0.7568 | 0.7580 | LLM Emb       | 1536      |
| DeBERTaV2XL    | 0.6805 | 0.6814 | BERT/DeBERTaV2   | 1536      |
| VoyageLarge2     | 0.7568 | 0.7580 | LLM Emb       | 1536      |
| OpenAI-TE3-large     | 0.7568 | 0.7580 | LLM Emb       | 1536      |

## Conclusion

## Acknowledgments

## How to cite
