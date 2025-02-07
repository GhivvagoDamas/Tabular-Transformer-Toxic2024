# A Transformer-based Tabular Approach to Detect Toxic Comments


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
a binary version of the ToLD-Br dataset. The model achieved a 76% accuracy
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

> [!CAUTION]
> Use Python 3.10.12 and CUDA 12.2.140.

This work used Google Colab as a development environment, mainly using the T4 free GPU tier.
<!--- Using a decent GPU is heavily encouraged. --->

### Data

ToLD-Br dataset is available at [here](https://github.com/JAugusto97/ToLD-Br) and on [HuggingFace](https://huggingface.co/datasets/JAugusto97/told-br). <br/>
However, you can find the original ToLD-Br dataset files, splits, and our two-column (text,label) binary version [here](data/).

<!--- Download and move to your current directory utils.py file --->

### How to reproduce the experiments:

> [!IMPORTANT]
> An API Key is required to reproduce results with **OpenAI** and **VoyageAI** Embedding Models (EM).
<br/>

To reproduce the experiments on Google Colab, use the FT-Transformer Binary Text Classifier [notebook](scr/FT_Transformer_Binary_Text_Classifier.ipynb) and the indicated requirement file to install the dependencies.

```bash
!pip install -r requirements_colab.txt
```
<br/>

For conda-like environments copy the whole project and install the required dependencies.
```bash
!pip install -r requirements.txt
```

> [!NOTE]
> - For the VoyageAI *voyage-large-2* embedding model processing the first 50MM tokens is free (including other EM);
> - <p align="justify"> To generate embeddings from OpenAI models it is imperative to ensure that your account has sufficient balance. Generating embeddings through OpenAI's API incurs costs; therefore, verifying or adding funds to your account is necessary. This ensures uninterrupted access to the required computational resources and facilitates smooth experimentation. </p>

> [!TIP]
> Refer to [OpenAI API Pricing](https://openai.com/api/pricing/) or [here](compute_gpt_costs.py) to calculate OpenAI usage costs based on the dataset size.

## List of Embedding Models

| Alias   | Model Name | Source | Language Support | Model Type | Output Dim |
| :---          |:---           | :---            |:---             |:---           |:---       |
| BERTimbau     | neuralmind/bert-large-portuguese-cased     | HuggingFace | Monolingual | BERT     | 1024    |
| AlbertinaPTBR | PORTULAN/albertina-900m-portuguese-ptbr-encoder       | HuggingFace | Monolingual | BERT/DeBERTaV2      | 1536      |
| SBERTimbau    | rufimelo/bert-large-portuguese-cased-sts       | HuggingFace | Monolingual | SBERT       | 1024      |
| ME5Large    | intfloat/multilingual-e5-large       | HuggingFace | Multilingual | SBERT       | 1024      |
| DeBERTaV2XL     | microsoft/deberta-v2-xlarge-mnli       | HuggingFace | Multilingual | BERT/DeBERTaV2       | 1536      |
| VoyageLarge2     | voyage-large-2      | Voyage AI | Multilingual | LLM Emb       | 1536      |
| OpenAI-TE3-large     | text-embedding-3-large | OpenAI | Multilingual | LLM Emb       | 1536      |

## Results

| Text Embedder | Test F1 (macro avg) | Test Acc | 
| :---          |     :---:      |     :---:     |   
| BERTimbau     | 0.7316 | 0.7333 | 
| AlbertinaPTBR    | 0,7066 | 0.6906 | 
| SBERTimbau    | 0.7317 | 0.7319 | 
| ME5Large    | 0.7568 | 0.7580 | 
| DeBERTaV2XL    | 0.6805 | 0.6814 | 
| VoyageLarge2     | 0.7568 | 0.7580 | 
| OpenAI-TE3-large     | 0.7568 | 0.7643 | 

## Conclusion Key Points 
### **Novel Approach for Binary Text Classification**  
- Tailored for structured tabular data.  
- Achieves **76% accuracy** and **75% F1-score** using **OpenAI text-embedding-3-large**.  
- Effectively detects **hate speech and toxicity** without **fine-tuning** or **transfer learning**.  

### **Dataset Review (ToLD-Br)**  
- **Concerns** about validity and reliability.  
- Issues with **imbalance in annotator agreement** due to **bias, term resignification, and stylistic slur usage**.  

### **Tabular Deep Learning (TDL) Models**  
- **FT-Transformer** excels in multimodal and structured data learning.  
- **Minimal computational resources** required beyond GPU usage.  
- Highlights the **importance of TDL and Embedding Models** in toxicity detection.  

### **Future Work**  
- **Exploring advanced embedding models**: **BGE M3 (FlagEmbeddings), SBERT**.  
- **Adapting LLMs** (e.g., **Sabiá, Aya**) into **robust embedding models**.  
- Investigating **bias detection & mitigation**.  
- Enhancing performance via **contextual features, Contrastive Learning, and RAG**.
  
## 🏆 Recognition & Acknowledgments
This paper was a **Best Paper Nominee** at the **34th Brazilian Conference on Intelligent Systems (BRACIS 2024)** 🎉. 

I am deeply grateful for the support and collaboration of my research colleagues and my master's advisors.  
Your guidance, insights, and dedication were instrumental in shaping the depth and quality of this research, 
leading to a publication that truly reflects our hard work and commitment to excellence. 

I sincerely appreciate you being incredible research partners, sharing your knowledge and expertise,
and helping me improve my writing and research skills.    

## How to cite
If you use this work, please cite:

```bibtex
@inproceedings{damas2024transformer,
  title={A Transformer-Based Tabular Approach to Detect Toxic Comments},
  author={Damas, Ghivvago and Torres Anchi{\^e}ta, Rafael and Santos Moura, Raimundo and Ponte Machado, Vinicius},
  booktitle={Brazilian Conference on Intelligent Systems},
  pages={18--30},
  year={2024},
  organization={Springer}
}



