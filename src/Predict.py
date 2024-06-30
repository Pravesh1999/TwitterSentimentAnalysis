import torch
from src.data.make_dataset import *
from transformers import BertTokenizer
from src.models.utils import *
from src.models.cnn import CNNModel
from src.models.lstm import LSTMTextClassifier
from src.models.lstm_multihead import LSTMMultiHeadAttention
from src.models.mlp import MLPClassifier
from src.settings import *


def predict(input_text, model):
    input_text = [input_text]
    cleaned_tweet = CleanTweets(input_text).clean_tweet()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_text = tokenizer.encode_plus(
        cleaned_tweet,
        add_special_tokens=True,
        max_length=MAX_SEQ_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    token_ids, atten_masks = encoded_text['input_ids'].squeeze(0), encoded_text['attention_mask'].squeeze(0)
    vocab_size = tokenizer.vocab_size
    if model == 'CNN':
        model_params = read_config_file('src/hparams.yml', 'CNNModel')
        model_params['vocab_size'] = vocab_size
        model = CNNModel(**model_params)
        model.load_state_dict(torch.load('models/CNNModel'))
    elif model == 'LSTM':
        model_params = read_config_file('src/hparams.yml', 'LSTMTextClassifier')
        model_params['vocab_size'] = vocab_size
        model = LSTMTextClassifier(**model_params)
        model.load_state_dict(torch.load('models/LSTMTextClassifier'))
    elif model == 'MLP':
        model_params = read_config_file('src/hparams.yml', 'MLPClassifier')
        model_params['vocab_size'] = vocab_size
        model = MLPClassifier(**model_params)
        model.load_state_dict(torch.load('models/MLPClassifier'))
    else:
        model_params = read_config_file('src/hparams.yml', 'LSTMMultiHeadAttention')
        model_params['vocab_size'] = vocab_size
        model = LSTMMultiHeadAttention(**model_params)
        model.load_state_dict(torch.load('models/LSTMMultiHeadAttention'))
    
    model.to(DEVICE)
    token_ids = token_ids.to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(token_ids.unsqueeze(0))
        prediction = torch.argmax(output, dim=1).item()
    return prediction

    