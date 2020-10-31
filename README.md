# The Devil is in the Details: Evaluating Limitations of Transformer-based Methods for Granular Tasks

This is the code and the dataset for the paper titled 

>[The Devil is in the Details: Evaluating Limitations of Transformer-based Methods for Granular Tasks. Brihi Joshi, Leonardo Neves, Neil Shah, Francesco Barbieri](https://github.com/brihijoshi/granular-similarity-COLING-2020/)

accepted at [The 28th International Conference on Computational Linguistics (COLING’20)](https://coling2020.org/).

# About

![An example of a pair of articles that are similar on a Granular level.](https://github.com/brihijoshi/granular-similarity-COLING-2020/blob/main/granular_example.png)
Figure: An example pair of articles from the __News Dedup__ dataset: Both report the same news event, and are thus _similar on a granular level_; the colored text indicates fine-grained details associated with this determination.  Both articles are also of the "sports" topic, and are thus _similar on an abstract level_.


Contextual embeddings derived from transformer-based neural language models have shown state-of-the-art performance for various tasks such as question answering, sentiment analysis, and textual similarity in recent years. Extensive work shows how accurately such models can represent _abstract_, semantic information present in text. In this expository work, we explore a tangent direction and analyze such models' performance on tasks that require a more _granular_ level of representation.  We focus on the problem of textual similarity from two perspectives: matching documents on a granular level (requiring embeddings to capture fine-grained attributes in the text), and an abstract level (requiring  embeddings to capture overall textual semantics). We empirically demonstrate, across two 
datasets from different domains, that despite high performance in abstract document matching as expected, contextual embeddings are consistently (and at times, vastly) outperformed by simple baselines like TF-IDF for more granular tasks. We then propose a simple but effective method to incorporate TF-IDF into models that use contextual embeddings, achieving relative improvements of up to 36% on granular tasks.


# License

Copyright (c) Snap Inc. 2020. 
This sample code is made available by Snap Inc. for informational purposes only. It is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability, fitness for a particular purpose, or non-infringement.  In no event will Snap Inc. be liable for any damages arising from the sample code or your use thereof.


# Quick Start

## Requirements

- Python 3.5.x

To install the dependencies used in the code, you can use the __requirements.txt__ file as follows -

```
pip install -r requirements.txt
```

### Installing baselines and datasets

1. Using the SIF Baseline, follow the installation steps given [here](https://github.com/PrincetonML/SIF) and add it to the ```code/SIF``` location.
1. For accessing the Bugrepo dataset, download the dataset from this [LogPAI Bugrepo repository.](https://github.com/logpai/bugrepo).

## Running the code

The code is organised as follows. 

```
├── code
│   ├── utils/ # This folder contains all the necessary pre-processing and skeleton code for the models. 
│   ├── SIF/ # This folder contains the SIF baseline requirements, installed as per the above instructions.
│   ├── news_dedup_experiments/ # This folder contains the experiments done with the News Dedup dataset
│   └── bugrepo_experiments/ # This folder contains the experiments done with the Bugrepo dataset
└── README.md
```

To run the code for specific experiments, go to their respective Jupyter Notebook and run the cells to train the models. 

For example, to run the code for the TFIDF Experiments for the Bugrepo dataset run the following - 

```
cd bugrepo_experiments/
jupyter notebook
```

and open the ```tf_idf_classification_bugrepo.ipynb``` notebook.


# Contact

If you face any problem in running this code, you can contact us at brihi16142\[at\]iiitd\[dot\]ac\[dot\]in or make an Issue in this repository.

For license information, see [LICENSE](LICENSE)

