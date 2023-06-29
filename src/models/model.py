import torch
from torch import nn


class BiLSTM(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, lstm_layers, emb_dropout, lstm_dropout,
                 fc_dropout, word_pad_idx):

        """
        Initializing the BiLSTM model which will be used in training.
        Args:
            input_dim: Input dimension of the model
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension of the model
            output_dim: Output dimension
            lstm_layers: Number of LSTM layers in the model
            emb_dropout: Embedding layer dropout rate
            lstm_dropout: LSTM layer dropout rate
            fc_dropout: Fully connected layer dropout rate
            word_pad_idx: ID of the padded token

        returns: None
        """

        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=word_pad_idx
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=lstm_dropout
        )

        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, sentence):
        embedding_out = self.emb_dropout(self.embedding(sentence))  # output: (sen_length*batch_size*embedding dim)

        lstm_out, _ = self.lstm(embedding_out)  # output: (sen_length*batch_size*(hidden_dim * 2))

        ner_out = self.fc(self.fc_dropout(lstm_out))  # output: (sen_length*batch_size*output_dim)
        return ner_out

    def init_weights(self):
        """
        Initializing the weights of the custom model

        Returns: None
        """
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def init_embeddings_w2v(self, word_pad_idx, pretrained=None, freeze=True):
        """
        Initializing the model with pretrained word2vec embedding vectors
        Args:
            word_pad_idx: ID of the padded token
            pretrained: Pretrained vectors of word embeddings
            freeze: Whether freeze the trained layers or not

        Returns: None

        """
        print("Initializing pretrained word2vec embedding")
        # initialize embedding for padding as zero
        self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)
        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.as_tensor(pretrained),
                padding_idx=word_pad_idx,
                freeze=freeze
            )

    def init_embeddings(self, word_pad_idx):
        """
        Initializing the model with embeddings
        Args:
            word_pad_idx: ID of the padded token

        Returns: None

        """
        print("initializing embedding")
        self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)

    def count_parameters(self):
        """
        Parameter counts of the custom build model
        Returns: Parameter counts
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
