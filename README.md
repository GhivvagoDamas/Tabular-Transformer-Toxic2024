# A Transformer-based Tabular Approach to Detect Toxic Comments

---
## Abstract
<p align="justify"> In recent years, there has been a significant increase in toxic
and hateful speech on social media platforms, becoming deeply entrenched
in online interactions. This issue has drawn the attention of researchers
from various academic fields, leading them to extend their focus to include
disciplines such as Natural Language Processing (NLP), Machine Learning,
and Linguistics, in addition to traditional areas like Law, Sociology,
Psychology, and Politics. This study introduces an approach for
detecting toxic and hateful speech on social media using Tabular Deep
Learning and NLP. The goal is to apply and enhance the accuracy of the
FT-Transformer model in detecting hateful and toxic content in textual
comments on social media in Brazilian Portuguese. An important aspect of
this research involves using language models and modern embedding models 
as language embedders, and evaluating their performance with the FT-Transformer,
a transformer-based tabular model. The experimental scenario uses
a binary version of the ToLD-Br dataset. The model achieved a 75% accuracy
rate and a 75% macro F1-score using the OpenAI text-embedding-3-large model.
While this approach showed strong performance, further improvements can be
achieved by incorporating additional features and methods to enhance the quality
of the embedding generated, especially for social media platforms. </p>

**Keywords:** Toxic and Hateful Speech · Deep Learning · FT-Transformer
· Embedding models · Text Classification.

<!--- <p align="justify"> This repository contains the code and resources for detecting toxic and hateful speech on social media, focusing 
on Brazilian Portuguese comments. The approach utilizes the Tabular Deep Learning model FT-Transformer and various embedding models 
to enhance the detection accuracy of toxic content. </p> --->
---
## Requeriments

Make sure to use Python Python 3.10.12 and CUDA 
<!---
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0 --->

This work used Google Colab as an environment, mainly using the T4 free GPU tier.
<!--- Using a decent GPU is heavily encouraged. --->

Install required Python dependencies.

```bash
!pip install -r requirements.txt
```
Optional python packages:

Dataset

ToLD-Br dataset is available at [here](https://github.com/JAugusto97/ToLD-Br) and on [HuggingFace](https://huggingface.co/datasets/JAugusto97/told-br). 
However, you can find the original ToLD-Br dataset files, splits, and our two-column (text,label) binary version [here](data/).

<!--- Download and move to your current directory utils.py file --->

### Running the experiment
<!---
```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
``` --->

To run the experiment use the Python Notebook available xxxxxx.

> [!IMPORTANT]
> An API Key is required to reproduce results with **OpenAI** and **VoyageAI** Embedding Models (EM).

> [!NOTE]
> For the VoyageAI *voyage-large-2* embedding model processing the first 50MM tokens is free (including other EM);
> <p align="justify"> To generate embeddings from OpenAI models it is imperative to ensure that your account has sufficient balance. Generating embeddings
  through OpenAI's API incurs costs; therefore, verifying or adding funds to your account is necessary. This ensures uninterrupted access to the required
  computational resources and facilitates smooth experimentation. </p>

> [!TIP]
> Refer to [OpenAI API Pricing]([https://github.com/JAugusto97/ToLD-Br](https://openai.com/api/pricing/)) or [here](compute_gpt_costs.py) to calculate OpenAI Embedding Models usage costs based on the dataset size.

## List of Embedding Models

| Alias   | Model Name | Source | Language Support | Model Type | Output Dim |
| :---          |:---           | :---            |        :---:         |     :---:      |          ---: |
| BERTimbau     | neuralmind/bert-large-portuguese-cased     | HuggingFace | Monolingual | BERT     | 1024    |
| AlbertinaPTBR | PORTULAN/albertina-900m-portuguese-ptbr-encoder       | HuggingFace | Monolingual | BERT/DeBERTaV2      | 1536      |
| SBERTimbau    | rufimelo/bert-large-portuguese-cased-sts       | HuggingFace | Monolingual | SBERT       | 1024      |
| ME5Large    | intfloat/multilingual-e5-large       | HuggingFace | Multilingual | SBERT       | 1024      |
| DeBERTaV2XL     | microsoft/deberta-v2-xlarge-mnli       | HuggingFace | Multilingual | BERT/DeBERTaV2       | 1536      |
| VoyageLarge2     | voyage-large-2      | Voyage AI | Multilingual | LLM Emb       | 1536      |
| OpenAI-TE3-large     | text-embedding-3-large | OpenAI | Multilingual | LLM Emb       | 1536      |

## Steps

![Alt text](FTT_approach.pdf)

## Results

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



> [!WARNING]
> Urgent info that needs immediate user attention to avoid problems.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.
