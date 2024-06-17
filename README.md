# A Transformer-based Tabular Approach to Detect Toxic Comments

---

<p align="justify"> This repository contains the code and resources for detecting toxic and hateful speech on social media, focusing 
on Brazilian Portuguese comments. The approach utilizes the Tabular Deep Learning model FT-Transformer and various embedding models 
to enhance the detection accuracy of toxic content. </p>

## Requeriments

Make sure to install all the packages
pip install tiktoken
pip install openai -U
pip install voyageai
pip install torch transformers
pip install sentencepiece
!pip install pytorch_frame[full]
```bash
pip install -U sentence-transformers
```


To reproduce results with **OpenAI** and **VoyageAI** Embedding Models (EM) an API Key is required.

For the VoyageAI *voyage-large-2* embedding model processing the first 50MM tokens is free (including other EM);
In the case of OpenAI, some credits need to be altho
check the usage cost of the model or refer to xxxxx to calculate the cost based on your dataset size.



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
