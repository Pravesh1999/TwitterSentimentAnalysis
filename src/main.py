import argparse
import os
import torch
import torch.nn as nn
from src.models.utils import *
from src.DatasetLoader import LoadDataset
from src.data.make_dataset import create_dataset
from src.settings import *
from src.models.train_model import *
from transformers import BertTokenizer, BertModel
from src.inference import *
from src.models.cnn import CNNModel
from src.models.lstm import LSTMTextClassifier
from src.models.lstm_multihead import LSTMMultiHeadAttention
from src.models.mlp import MLPClassifier
from src.Predict import predict

def load_data(json_path):
    """
    Function to read json file
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise Exception(f"The file {json_path} was not found.")
    except json.JSONDecodeError:
        raise Exception(f"The file {json_path} could not be decoded.")

def configure_model(args, vocab_size):
    model_class = {
        'CNNModel': CNNModel,
        'LSTMTextClassifier': LSTMTextClassifier,
        'MLPClassifier': MLPClassifier,
        'LSTMMultiHeadAttention': LSTMMultiHeadAttention
    }.get(args.model_name, MLPClassifier)
    
    model_params = read_config_file('src/hparams.yml', args.model_name)
    model_params['vocab_size'] = vocab_size
    set_seed(2023)
    model, optimizer = init_model(model_class, model_params)
    return model, optimizer

def main():
    set_seed(2023)
    parser = argparse.ArgumentParser(description='Main Training Script for Sentiment Analysis')
    parser.add_argument('--model_name', type=str, help='Name of the model you want to load')
    parser.add_argument('--load_and_clean', action='store_true', help='Load and clean dataset')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train-model', action='store_true', help='Train model')
    group.add_argument('--infer-model', action='store_true', help='Infer model')
    group.add_argument('--predict', action='store_true', help='Make Predictions')
    args = parser.parse_args()

    if args.load_and_clean:
        create_dataset()

    if args.train_model:
        set_seed(2023)
        data = load_data(os.path.join(PREPROCESSED_DATA_PATH, 'split_data.json'))
        train_texts, train_labels = data["train_texts"], data["train_labels"]
        valid_texts, valid_labels = data["valid_texts"], data["valid_labels"]
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = LoadDataset(train_texts, train_labels, tokenizer)
        valid_dataset = LoadDataset(valid_texts, valid_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
        model, optimizer = configure_model(args, tokenizer.vocab_size)
        loss_fn = nn.CrossEntropyLoss()
        model.to(DEVICE)
        early_stopping = EarlyStopping(patience=3, verbose=True)
        train(model, optimizer, train_loader, valid_loader, loss_fn, epochs=50, model_save_path=f'models/{args.model_name}', early_stopping=early_stopping)

    elif args.infer_model:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model, _ = configure_model(args, tokenizer.vocab_size)
        try:
            # Load the trained model for inference
            model.load_state_dict(torch.load(f'models/{args.model_name}'))
            model.to(DEVICE)
            loss_fn = nn.CrossEntropyLoss()
            make_inference(model, args.model_name, loss_fn, f'{PREPROCESSED_DATA_PATH}/split_data.json')
        except FileNotFoundError:
            raise Exception("Model file not found. Ensure the model has been trained and saved.")
        
    elif args.predict:
        input_text = input('Enter the tweet for sentiment analysis: ')
        model_name = input('Enter the model you want to use for prediction: ')
        predictions = predict(input_text, model_name)
        pred_dict = {0: 'Angry', 1:'Disappointed', 2:'Happy'}
        model_pred = pred_dict[int(predictions)]




if __name__ == '__main__':
    main()
