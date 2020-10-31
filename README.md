# The Devil is in the Details: Evaluating Limitations of Transformer-based Methods for Granular Tasks

This is the code and the dataset for the paper titled 

>[The Devil is in the Details: Evaluating Limitations of Transformer-based Methods for Granular Tasks. Brihi Joshi, Leonardo Neves, Neil Shah, Francesco Barbieri](https://github.com/brihijoshi/granular-similarity-COLING-2020/)

accepted at [The 28th International Conference on Computational Linguistics (COLING’20)](https://coling2020.org/).

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

### Additional Resources

For running the experiments - 
1. Using the SIF Baseline, follow the installation steps given [here](https://github.com/PrincetonML/SIF).
1. For accessing the Bugrepo dataset, download the dataset from this [LogPAI Bugrepo repository.](https://github.com/logpai/bugrepo)

## Running the code

The code is organised as follows. 

```
code/
-----------utils/ # This folder contains all the necessary pre-processing and skeleton code for the models. 
-----------SIF/ # This folder contains the SIF baseline requirements
-----------news_dedup_experiments/ # This folder contains the experiments done with the News Dedup dataset
-----------bugrepo_experiments/ # This folder contains the experiments done with the Bugrepo dataset
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

