import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) for sentiment analysis using LSTM layers.
    """
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the SentimentRNN model.

        Args:
        - vocab_size (int): The number of words in the vocabulary.
        - output_size (int): The number of output classes.
        - embedding_dim (int): Size of the word embedding vectors.
        - hidden_dim (int): Number of features in the LSTM hidden state.
        - n_layers (int): Number of LSTM layers.
        - drop_prob (float): optional (default=0.5) Dropout probability.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        """
        Perform a forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        - hidden (tuple): The initial hidden and cell states of the LSTM.

        Returns:
        - out (torch.Tensor): Output tensor containing the predictions of the model.
        - hidden (tuple): Updated hidden and cell states.
        """
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out[:, -1, :]  
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        """
        Initialize the hidden state and cell state of the LSTM.

        Args:
        - batch_size (int): The batch size of the input data.
        - device: cuda if available or cpu

        Returns:
        - hidden (tuple): A tuple containing the initialized hidden and cell states.
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden