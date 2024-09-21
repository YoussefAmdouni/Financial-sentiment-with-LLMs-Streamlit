# src/trainLstm.py
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import argparse
import torch
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from lstmclf import SentimentRNN
from utils import process, create_vocab, map_to_int, text_to_int, dataloader, save_model


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


def validation(model, criterion, x_test, y_test, batch_size, seq_length, device):
    test_loss = []
    correct_count = 0  
    total_count = 0
    for inputs, labels in dataloader(x_test, y_test, batch_size, seq_length, shuffle=False):

        h = model.init_hidden(inputs.size(0), device)
        inputs, labels = inputs.to(device), labels.to(device)
            
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
        loss = criterion(output, labels.long())

        test_loss.append(loss.item())

        
        _, preds = torch.max(output, 1)
        correct_count += (preds == labels).sum().item()
        total_count += labels.size(0)

    test_accuracy = correct_count / total_count 

    return np.mean(test_loss), test_accuracy


def train(model, criterion, optimizer, x_train, y_train, x_valid, y_valid, device, batch_size, epochs, seq_length, saving_path, clip=5):
    
    model.train()
    
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], [] 
    best_accuracy = 0.0

    for e in range(epochs):
        train_loss = []
        correct_count = 0  
        total_count = 0
        for inputs, labels in dataloader(x_train, y_train, batch_size, seq_length, shuffle=True):
            h = model.init_hidden(inputs.size(0), device)
            inputs, labels = inputs.to(device), labels.to(device)
            
            h = tuple([each.data for each in h])
            
            model.zero_grad()
            output, h = model(inputs, h)

            loss = criterion(output, labels.long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            train_loss.append(loss.item())

            _, preds = torch.max(output, 1)  
            correct_count += (preds == labels).sum().item()
  
            total_count += labels.size(0) 

        else:
            model.eval()            
            with torch.no_grad():
                test_loss, test_accuracy = validation(model, criterion, x_valid, y_valid, batch_size, seq_length, device) 

        epoch_loss = np.mean(train_loss)
        epoch_accuracy = correct_count / total_count  

        print(f"Epoch: {e+1}/{epochs}...",
              f"Train Loss: {epoch_loss:.6f}",
              f"Train Accuracy: {epoch_accuracy:.6f}",
              f"Validation Loss: {test_loss:.6f}",
              f"Validation Accuracy: {test_accuracy:.6f}")

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_model(model, saving_path)
            print('best model saved!')

        model.train() 

    return train_accuracies, train_losses, test_accuracies, test_losses        

def save_plots(tracc, trlos, tsacc, tslos, filename):
    plt.figure(figsize=(12, 6))
    
    # Plot Training/Validation Losses
    plt.subplot(1, 2, 1)
    plt.title("Training/Validation Losses")
    plt.plot(trlos, label='Training loss')
    plt.plot(tslos, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Training/Validation Accuracies
    plt.subplot(1, 2, 2)
    plt.title("Training/Validation Accuracies")
    plt.plot(tracc, label='Training accuracy')
    plt.plot(tsacc, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save the figure
    plt.savefig(filename)
    plt.close()



def main(args):
    # Load data
    df = load_data(args.data_file)

    df['clean_sentence'] = df['sentence'].apply(lambda x: process(x))
    words = ' '.join(df['clean_sentence'].astype(str)).split()
    vocab = create_vocab(words)
    vocab_to_int = map_to_int(vocab)
    
    df['int_sentence'] = df['clean_sentence'].apply(lambda x: text_to_int(x, vocab_to_int))
    df = df[['int_sentence', 'target']]
    
    # Train-test split
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=123)
    X_train, y_train = df_train['int_sentence'].to_list(), df_train['target'].to_list()
    X_test, y_test = df_test['int_sentence'].to_list(), df_test['target'].to_list()

    vocab_size = len(vocab_to_int) + 1
    output_size = 3
    embedding_dim=300
    hidden_dim=256 
    batch_size = 512
    n_layers=2

    # Define model, loss, and optimizer
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    model = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    tracc, trlos, tsacc, tslos = train(model, criterion, optimizer, X_train, y_train, X_test, y_test, device, batch_size, args.epochs, args.max_len, args.saving_path)
    save_plots(tracc, trlos, tsacc, tslos, args.eval_path)

    model.load_state_dict(torch.load(args.saving_path))

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader(X_test, y_test, batch_size, args.max_len, shuffle=False):
            
            h = model.init_hidden(inputs.size(0), device)
            inputs, labels = inputs.to(device), labels.to(device)
                
            h = tuple([each.data for each in h])
            output, h = model(inputs, h)
            
            _, preds = torch.max(output, 1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(true_labels, predictions)

    mapper = {'positive': 2, 'neutral': 1, 'negative': 0}
    class_names = [key for key, value in sorted(mapper.items(), key=lambda item: item[1])]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix_lstm.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a BERT-based sentiment classifier.')
    parser.add_argument('--data_file', type=str, default='data/Sentences_75Agree.txt', help='Path to the data file.')
    parser.add_argument('--max_len', type=int, default=82, help='Max sequence length.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--saving_path', type=str, default='models/bestmodellstm.model', help='Path to save the best model.')
    parser.add_argument('--eval_path', type=str, default='results/model_eval.png', help='Path to training vs validation performances.')
    args = parser.parse_args()
    main(args)
