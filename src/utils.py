import yaml
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch

def read_config_file(file_name,model_name):
    """Reads the yaml file to extract hyperparameters"""
    try:
        with open(file_name, "r") as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
            if model_name:
                if model_name in data:
                    model_parameters = data[model_name]
            else:
                model_parameters = data
    except Exception as e:
        print(e)
        model_parameters = {}
    return model_parameters

def init_model(model_name, model_parameters, learning_rate=0.01, rho=0.95):
    """
    Initializes a model and its optimizer.
    
    Args:
    - model_name: The class name of the model to be initialized.
    - model_parameters (dict): Parameters to be passed to the model's constructor.
    - learning_rate (float): Learning rate for the optimizer.
    - rho (float): Rho parameter for RMSprop optimizer.
    
    Returns:
    Tuple of (model, optimizer).
    """
    model = model_name(**model_parameters)  # Initialize the model with given parameters
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=rho)  # Initialize the optimizer
    return model, optimizer

def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)