from torch import nn


class Discriminator(nn.Module):

    # DIS_ATTR = ['input_dim', 'dis_layers', 'dis_hidden_dim', 'dis_dropout']

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        """
        Discriminator initialization.
        """
        super(Discriminator, self).__init__()

        self.output_size = 2
        self.input_size = input_size
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        layers = []
        for i in range(self.n_layers + 1):
            if i == 0:
                input_size = self.input_size
            else:
                input_size = self.hidden_size
            output_size = self.hidden_size if i < self.n_layers else self.output_size
            layers.append(nn.Linear(input_size, output_size))
            if i < self.n_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, enc_out):
        return self.layers(enc_out)
