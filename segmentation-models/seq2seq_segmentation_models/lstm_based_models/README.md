Sequence to Sequence Models for isiZulu
===

Author: Sthembiso Mkhwanazi

Overview
This repository is dedicated to notebooks designed for training and testing sequence-to-sequence models, with a primary focus on segmenting and translating isiZulu.

Setup Instructions

To ensure all necessary packages are installed, please execute the following script in your terminal:

`./installs.sh`


Training LSTM Models

---
Configuration
For training a new model, create a JSON file in the config/ directory specifying the following keys:

MODEL_NAME
DATA_TRAIN
DATA_VALIDATION
DATA_TEST
SRC_LANGUAGE
TGT_LANGUAGE
SRC_TOKENIZER
BATCH_SIZE
LATENT_DIM
NUM_EPOCHS

Project Structure
The directory structure of the project is outlined below:

The file hirachy is as follows

```	
home
    └── data
        └── projects
            └── seq2seq
                ├── data
                │   ├── lstm_model
                │   └── config: {"config files"}
                └── models: {"place to save the models"}
                    ├── installs.sh  # Script for package installation
                    ├── train_lstm.ipynb  # Notebook for initial training
                    └── load_lstm.ipynb  # Notebook for model evaluation been created.
    					
```					
Data Division
The data files are categorized into three sections:

`DATA_TRAIN` for the training dataset
`DATA_VALIDATION` for the validation set
`DATA_TEST` for the testing set

Copy your configuration file to `config/config.json` to avoid modifying the notebooks for different datasets or parameters.
```
{
    "MODEL_NAME": "LSTM_segmenter",
    "DATA_TRAIN": "scramble2_train_set.csv",
    "DATA_VALIDATION": "scramble2_valid_set.csv",
    "DATA_TEST": "test_set.csv",
    "SRC_LANGUAGE": "tokens",
    "TGT_LANGUAGE": "segments",
    "BATCH_SIZE": 64,
    "LATENT_DIM": 256,
    "NUM_EPOCHS": 18
}
```

Important Note:

The test dataset can either be `test_set.csv`, common across all models, or `new_test_set.csv`, which excludes certain tokens not analyzable by two analyzers.
NOTE also: The Ukwabelana segmenter is trained with using the same data set file i.e `train_set.csv` the only thing differs is the TGT_LANGUAGE. 

load_lstm_model
---

The `load_lstm_model` notebook is designed to load an existing model and evaluate its performance against a saved test/evaluation dataset. It calculates the corpus BLEU and chrF scores for the model on the test set.

---

