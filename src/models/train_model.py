import torch.nn as nn
import torch.optim as optim
import torch
from src.settings import *
import time
import os
import numpy as np
from src.models.utils import *


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        
def train(model, optimizer, train_loader, val_loader, loss_fn, epochs=10, 
          model_save_path='../models/best_model.pth',early_stopping=None,device=DEVICE):
    set_seed(2023)
    best_accuracy = 0
    print("=======Starting Training========")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*60)
    for epoch in range(epochs):
        model.train()
        t0_epoch = time.time()
        total_loss = 0
        for batch_X ,batch_labels in train_loader:
            input_ids, attention_masks = batch_X
            # Zero the grads for every batch
            model.zero_grad()
            # Perform forward pass
            logits = model(input_ids.to(DEVICE))
            # Compute loss and accumulate the loss
            loss = loss_fn(logits, batch_labels.to(device))
            total_loss += loss.item()
            # Perform backward pass for the batch
            loss.backward()
            # Update parameters
            optimizer.step()
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluation on validation set
        if val_loader is not None:
            
            val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
            if early_stopping:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                
            # Calculate training time for each epoch
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
    print("\n")
    print("==========Best Accuracy After training================",best_accuracy)
    
def evaluate(model, val_loader, loss_fn, device=DEVICE):
    model.eval()  # Set the model to evaluation mode.
    val_accuracy = []
    val_loss = []
    with torch.no_grad():
        for val_batch_input, val_batch_label in val_loader:
            input_ids, attention_masks = val_batch_input
            # val_batch_embeddings = embedding_layer(input_ids.to(device))
            logits = model(input_ids.to(DEVICE))
            loss = loss_fn(logits, val_batch_label.to(device))
            val_loss.append(loss.item())

            #preds = torch.argmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            labels = torch.argmax(val_batch_label, dim=1)
            accuracy = (preds == labels.to(device)).float().mean() * 100
            val_accuracy.append(accuracy.item())

        val_loss = torch.tensor(val_loss).mean().item()
        val_accuracy = torch.tensor(val_accuracy).mean().item()
    
    return val_loss, val_accuracy
        
        
