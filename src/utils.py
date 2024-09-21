import torch
import random
import re
from collections import Counter

def process(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

def create_vocab(words):
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    return vocab

def map_to_int(vocab):
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab_to_int

def text_to_int(text, vocab_to_int):
    return [vocab_to_int[word] for word in text.split()]

def dataloader(messages, labels, sequence_length, batch_size, shuffle=False):
    if shuffle:
        indices = list(range(len(messages)))
        random.shuffle(indices)
        messages = [messages[i] for i in indices]
        labels = [labels[i] for i in indices]

    total_sequences = len(messages)
    
    for idx in range(0, total_sequences, batch_size):
        batch_messages = messages[idx: idx+batch_size]
        
        batch = torch.zeros((len(batch_messages), sequence_length), dtype=torch.int64)
        for batch_num, tokens in enumerate(batch_messages):
            token_tensor = torch.tensor(tokens)
            start_idx = max(sequence_length - len(token_tensor), 0)
            batch[batch_num, start_idx:] = token_tensor[:sequence_length]
        
        label_tensor = torch.tensor(labels[idx: idx+len(batch_messages)])
        yield batch, label_tensor

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model