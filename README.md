# WISMIR3: A Multi-Modal Dataset to Challenge Text-Image Retrieval Approaches
This repository contains the code and data of the WISMIR3 Dataset.

[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-md-dark.svg)](https://openreview.net/forum?id=Q93yqpfECQ)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/floschne/wismir3)

### Abstract 
This paper presents WISMIR3, a multi-modal dataset comprising roughly 300K text-image pairs from Wikipedia. With a sophisticated automatic ETL pipeline, we scraped, filtered, and transformed the data so that WISMIR3 intrinsically differs from other popular text-image datasets like COCO and Flickr30k. We prove this difference by comparing various linguistic statistics between the three datasets computed using the pipeline. The primary purpose of WISMIR3 is to use it as a benchmark to challenge state-of-the-art text-image retrieval approaches, which already reach around 90% Recall@5 scores on the mentioned popular datasets. Therefore, we ran several text-image retrieval experiments on our dataset using current models, which show that the models, in fact, perform significantly worse compared to evaluation results on COCO and Flickr30k. In addition, for each text-image pair, we release features computed by Faster-R-CNN and CLIP models. With this, we want to ease and motivate the use of the dataset for other researchers.

## WISMIR ETL Pipeline

A simple and efficient ETL Pipeline to access and transform the WikiCaps dataset.

The  tool is capable of
- collecting linguistic metadata based on the captions using different models and frameworks
- flexibly filtering the data with user-defined filters
- downloading the images in parallel
- applying customizable transformations to the images and text after the download
- persisting the data in an easy to use and efficient format

### How to run the pipeline

Requirements:
`conda env create --file environment.yml`

```python
python main.py --config=configs/config_gpu_server_spacy_v2.yml
```
### Results DataFrame Columns

The ETL Pipeline outputs a Pandas DataFrame containing metadata about the captions and links to the images.

### How to read the DataFrame

Requirements:
`pip install pandas pyarrow`

```python
import pandas as pd

metadata = pd.read_feather("path/to/metadata.feather", use_threads=True)
```

| ColumnId			| Description																| Datatype	|
|-------------------|---------------------------------------------------------------------------|-----------|
| wikicaps_id		| ID (line number) of the row in the original WikiCaps Dataset __img_en__ 	| int		|
| wikimedia_file    | Wikimedia File ID of the Image associated with the Caption				| str		|
| caption			| Caption of the Image														| str		|
| image_path		| Local path to the (downloaded) image										| str		|
| num_tok			| Number of Tokens in the caption											| int		|
| num_sent			| Number of Sentences in the caption										| int		|
| min_sent_len		| Minimum number of Tokens in the Sentences of the caption					| int		|
| max_sent_len		| Maximum number of Tokens in the Sentences of the caption					| int		|
| num_ne			| Number of Named Entities in the caption									| int		|
| num_nouns			| Number of Tokens with NOUN POS Tag **										| int		|
| num_propn			| Number of Tokens with PROPN POS Tag **									| int		|
| num_conj			| Number of Tokens with CONJ POS Tag **										| int		|
| num_verb			| Number of Tokens with VERB POS Tag **										| int		|
| num_sym			| Number of Tokens with SYM POS Tag **										| int		|
| num_num			| Number of Tokens with NUM POS Tag **										| int		|
| num_adp			| Number of Tokens with ADP POS Tag **										| int		|
| num_adj			| Number of Tokens with ADJ POS Tag **										| int		|
| ratio_ne_tok		| Ratio of tokens associated with Named Entities vs all Tokens **			| int		|
| ratio_noun_tok	| Ratio of tokens tagged as NOUN vs all Tokens **							| int		|
| ratio_propn_tok	| Ratio of tokens tagged as PROPN vs all Tokens **							| int		|
| ratio_all_noun_tok| Ratio of tokens tagged as PROPN or NOUN vs all Tokens **					| int		|
| fk_re_score		| Flesch-Kincaid Reading Ease score of the Caption ***						| int		|
| fk_gl_score		| Flesch-Kincaid Grade Level score of the Caption ***						| int		|
| dc_score			| Dale-Chall score of the Caption ***										| int		|
| ne_texts			| Surface form of detected NamedEntities									| List[str]	|
| ne_types			| Types of the detected NamedEntities (PER, LOC, GPE, etc.)					| List[str]	|

** This column is only available if `config.input_data.pos_tag_stats == True`
. [Click here](https://universaldependencies.org/docs/u/pos/) for a detailed description of the POS Tags

*** This column is only available if `config.input_data.readability_scores == True`
. [Click here](https://en.wikipedia.org/wiki/List_of_readability_tests_and_formulas) for more information about
Readability Scores

## Citation
If you use this dataset or code, please cite:
```bibtex
@inproceedings{
schneider2024wismir,
title={{WISMIR}3: A Multi-Modal Dataset to Challenge Text-Image Retrieval Approaches},
author={Florian Schneider and Chris Biemann},
booktitle={3rd Workshop on Advances in Language and Vision Research (ALVR)},
year={2024},
url={https://openreview.net/forum?id=Q93yqpfECQ}
}
```

## WikiCaps publication
For more Information about the original WikiCaps Dataset, see [https://www.cl.uni-heidelberg.de/statnlpgroup/wikicaps/](https://www.cl.uni-heidelberg.de/statnlpgroup/wikicaps/)

```
Shigehiko Schamoni, Julian Hitschler and Stefan Riezler
A Dataset and Reranking Method for Multimodal MT of User-Generated Image Captions
Proceedings of the 13th biennial conference of the Association for Machine Translation in the Americas (AMTA), Boston, MA, USA, 2018
```

