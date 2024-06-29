Twitter Sentiment Analysis with Deep Learning
==============================

This project applies deep learning techniques to analyze sentiments expressed in tweets. It uses four different models implemented in PyTorch: MLP, CNN, BiLSTM, and BiLSTM with Multi-Head Attention. Each model aims to effectively capture the contextual nuances of Twitter data. The source code of the project is present in the src directory.

## Models Overview:

1. MLPClassifier:
    - Input: (EMBEDDING_DIM*SEQUENCE_LENGTH, 2048)
    - First Hidden Layer: (2048, 1024)
    - Second Hidden Layer: (1024, 612)
    - Final Layer: (612, 3)
2. CNNModel:
    - filters:  Sizes of 3, 4, 5 with filter counts of 32, 64, 128 respectively
    - Pooling: Max pooling after each convolution
    - Output Layer: Sum of filters to 3 units

3. LSTMTextClassifier
    - Configuration: Bidirectional with 6 layers
    - Hidden State: 128 units
    - Output: Fully connected layer

4. LSTMMultiHeadAttention:
    - Configuration: Bidirectional with 6 layers and 8 attention heads
    - Hidden State: 128 units
    - Output: Combined attention heads passed through a fully connected layer
   

## How to run the project:

### Prerequisites:
Before running the project, ensure you have Python installed on your system. Then, install all required Python packages using the following command in the root directory of the project:
'''bash
conda create -n myenv python=3.x
conda activate myenv
conda install pip
pip install -r requirements.txt
'''
### Load and Clean the dataset:

This is the first step to load the data and preparing it for modelling. Run the following command first:
'''bash
python -m src.main --load_and_clean
'''
### Training individual models:
Type in the following command to train MLP model:
'''bash
 python -m src.main --model_name MLPClassifier --train-model
'''
You can replace this with any model you want to train:
The following are the available model_name arguments:
1. MLPClassifier
2. CNNModel
3. LSTMTextClassifier
4. LSTMMultiHeadAttention

Training these models on a large dataset may take between 1 to 34 hours depending on the model complexity and the hardware used.

### Testing the model:

Once the model has be trained, it's performance can be evaluated using the test set. 

From the root project folder run the following command:
'''bash
 python -m src.main --model_name MLPClassifier --infer-model
'''
 replace the model name with the model inference you want to visualize. The model_name will have the same name as the one you trained.


This README provides clear instructions on how to install dependencies, how to run different components of the project, and how to contribute to the project, tailored for users familiar with GitHub projects.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
