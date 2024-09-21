# src/trainBert.py
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from bertpluslayers import BertSentimentClassifier

# Fix Seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def load_data(file_path):
    df = pd.read_csv(file_path, sep="@", encoding='latin-1', header=None)
    df.rename(columns={0: 'sentence', 1: 'target'}, inplace=True)
    mapper = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['target'] = df['target'].apply(lambda x: mapper[x])
    return df

def tokenize_data(tokenizer, sentences, max_len):
    return tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=max_len,
        return_tensors='pt'
    )

def train_bert_model(model, data_loader, criterion, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0
    for data in data_loader:
        input_ids = data[0].to(device)
        attention_mask = data[1].to(device)
        targets = data[2].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.max(outputs, dim=1)[1]
        loss = criterion(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_bert_model(model, data_loader, criterion, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for data in data_loader:
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            targets = data[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.max(outputs, dim=1)[1]
            loss = criterion(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, np.mean(losses)

def prediction(model, sentence, tokenizer, device, max_len):
        model.eval()
        encoding = tokenizer.encode_plus(
            sentence,
            max_length=max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        return output.detach().cpu().numpy().argmax()

def main(args):
    # Load data
    df = load_data(args.data_file)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    # Data split
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=seed, stratify=df.target.values)
    
    # Tokenize data
    encoded_data_train = tokenize_data(tokenizer, df_train.sentence.values, args.max_len)
    encoded_data_val = tokenize_data(tokenizer, df_test.sentence.values, args.max_len)
    
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df_train.target.values)
    
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df_test.target.values)
    
    # Data loaders
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    dataloader_train = DataLoader(dataset_train, batch_size=8)
    dataloader_val = DataLoader(dataset_val, batch_size=32)
    
    # Initialize model, optimizer, and scheduler
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model = BertSentimentClassifier(3, args.hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(dataloader_train) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    history = defaultdict(list)
    best_accuracy = 0
    
    for e in range(args.epochs):
        print(f'Epoch {e + 1}/{args.epochs}')
        train_acc, train_loss = train_bert_model(model, dataloader_train, criterion, optimizer, device, scheduler, len(df_train))
        val_acc, val_loss = eval_bert_model(model, dataloader_val, criterion, device, len(df_test))
        
        print(f'Train loss: {train_loss} Train accuracy: {train_acc} Val loss: {val_loss} Val accuracy: {val_acc}\n')
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        if best_accuracy < val_acc:
            best_accuracy = val_acc
            torch.save(model.state_dict(), args.saving_path)
            print('Best model saved!!')
    
    
    valpreds = [prediction(model, i, tokenizer, device, args.max_len) for i in df_test.sentence.values]
    cm = confusion_matrix(df_test.target.values, valpreds)
    class_names = ['negative', 'neutral', 'positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix_bert.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a BERT-based sentiment classifier.')
    parser.add_argument('--data_file', type=str, default='data/Sentences_75Agree.txt', help='Path to the data file.')
    parser.add_argument('--max_len', type=int, default=82, help='Max sequence length.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Linear layer hidden dimensions.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--saving_path', type=str, default='models/bestmodel.model', help='Path to save the best model.')
    args = parser.parse_args()
    main(args)