import torch 
from torch import nn
from transformers import BertModel

class BertSentimentClassifier(nn.Module):   
    """
    A sentiment classifier using a pre-trained BERT model followed by two fully connected layers.
    """ 
    def __init__(self, n_classes, hidden_dim):
        """
        Initializes the SentimentClassifier model by loading a pre-trained BERT model and defining additional layers.

        Args:
        - n_classes (int): The number of output classes.
        - hidden_dim (int): Size of the hidden layer.
        """
        super(BertSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
        - input_ids (tensor): Tensor of input token IDs for the BERT model.
        - attention_mask (tensor): Tensor indicating which tokens should be attended to (1 for valid tokens, 0 for padding).
        
        Returns:
        - logits (tensor): The raw, unnormalized predictions (logits) from the final layer, which represent class scores.
        """
        model_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = model_state['pooler_output']
        fc = self.drop(pooled_output)
        return self.fc2(self.fc1(fc))