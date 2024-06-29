import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import random
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.settings import *
import yaml
import json
from src.models.utils import *
from src.DatasetLoader import *
import argparse
import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import plotly.graph_objects as go


def inference(model, model_name,test_loader, loss_fn, device=DEVICE):
    """
    Function to make inference of the trained model. The function will display the classification report and also 
    the accuracy of each of the model
    """
    model.eval()  # Set the model to evaluation mode.
    total_loss, total_correct = 0, 0
    total_samples = 0
    all_preds, all_labels = [], []
    heads_count = 8
    attention_weights_per_head = [[] for _ in range(heads_count)]
    with torch.no_grad():
        for test_batch_input, test_batch_label in test_loader:
            batch_attention_weights = []
            input_ids, attention_masks = [t.to(device) for t in test_batch_input]
            test_batch_label = test_batch_label.to(device)

            logits = model(input_ids)
            loss = loss_fn(logits, test_batch_label)
            total_loss += loss.item() * test_batch_label.size(0)

            preds = torch.argmax(logits, dim=1)
            test_batch_labels = torch.argmax(test_batch_label, dim=1)
            total_correct += (preds == test_batch_labels).sum().item()
            total_samples += test_batch_label.size(0)

            # Collect all predictions and labels to use in the confusion matrix and classification report
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(test_batch_labels.cpu().numpy())
            # Store attention weights from all the batches for lstm with attention layer
            # if model_name == "LSTMMultiHeadAttention":
            #     if model.last_attention_weights:
            #         for i, weights in enumerate(model.last_attention_weights):
            #             attention_weights_per_head[i].append(weights.cpu())
            #     #average_attention_weights = [torch.cat(weights_list, dim=0).mean(dim=0) for weights_list in attention_weights_per_head]
            #     average_attention_weights = []
            # else:
            #     average_attention_weights = []     
    avg_loss = total_loss / total_samples
    avg_accuracy = (total_correct / total_samples) * 100

    # Compute confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, digits=4, target_names=['angry', 'disappointed', 'happy'])  # Adjust class names as necessary
    average_attention_weights = []
    return avg_loss, avg_accuracy, average_attention_weights, cm, cr


def plot_attention_heatmap(tokens, attention_weights, cmap='viridis'):
    # Assuming attention_weights is a list of numpy arrays
    num_heads = len(attention_weights)
    fig, axs = plt.subplots(1, num_heads, figsize=(30, 10))

    for i, ax in enumerate(axs):
        # Convert to numpy array and squeeze dimensions
        weights = attention_weights[i].cpu().squeeze().numpy()
        
        # Normalize weights for visualization purposes
        normalized_weights = weights / np.max(weights)
        
        # Bar chart for attention weights
        ax.bar(range(len(tokens)), normalized_weights, alpha=0.7)
        
        # Set x-axis labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        
        # Set y-axis label and title for each subplot
        ax.set_ylabel('Attention Weights')
        ax.set_title(f'Head {i+1}')
        
        # Set the same y-axis limit for comparison across heads
        ax.set_ylim(0, 1)
        
    plt.tight_layout()
    plt.show()





def make_inference(model, model_name, loss_fn, file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_texts, test_labels = data['test_texts'], data['test_labels']
        
    except FileNotFoundError:
        raise Exception(f"The file {file_name} was not found.")
    except json.JSONDecodeError:
        raise Exception(f"The file {file_name} could not be decoded.")
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")
    # Initialize BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load the test dataset
    test_dataset = LoadDataset(test_texts, test_labels,tokenizer)
    # Get the number of records in test dataset
    print("Number of records in test set is: ", len(test_dataset))
    # Get the number of CPU cores
    num_cpus = os.cpu_count()
    # Get the data into batches
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,num_workers=num_cpus)
    os.makedirs(os.path.dirname(INFERENCE_FILE_PATH), exist_ok=True)
    # Make inference on the model parameters        
    _, avg_accuracy, attention_weights,cm, cr = inference(model, model_name, test_loader, loss_fn, DEVICE)
    print("Accuracy: ", avg_accuracy)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(INFERENCE_FILE_PATH,model_name+'confusion_matrix_cnn.png'))
    plt.show()
    print("Classification Report:\n", cr)
    
    # if model_name == 'LSTMMultiHeadAttention':
    #     attention_weights = [x.to('cpu') for x in attention_weights]
    #     print(test_texts[3],test_labels[3])
    #     inputs = tokenizer(test_texts[3], return_tensors="pt", padding='max_length', truncation=True, max_length=120,return_attention_mask=True)

    #     # Assuming 'attention_weights' is a tensor of shape [1, seq_length, 1]
    #     # and 'inputs' includes 'input_ids' used in tokenization.
    #     attention = attention_weights[4].cpu().squeeze().numpy()  # Reduce dimensions
    #     token_ids = inputs['input_ids'].squeeze().tolist()  # Squeeze out batch dimension if present
    #     tokens = tokenizer.convert_ids_to_tokens(token_ids)

    #     # plt.figure(figsize=(40, 5))
    #     # plt.bar(range(len(tokens)), attention, alpha=0.7)
    #     # plt.xlabel('Token Index')
    #     # plt.ylabel('Attention Weights')
    #     # plt.title('Attention Weights Visualization for the First Sequence')
    #     # plt.xticks(range(len(tokens)), tokens, rotation=90)
    #     # plt.show()

    #     plot_attention_heatmap(tokens, attention_weights)



        
        
        
        
    
    