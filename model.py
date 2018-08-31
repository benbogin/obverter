import torch
from torch import nn
from torch.nn.functional import log_softmax, softmax


def get_sentence_lengths(comm_input, vocab_size):
    return torch.sum(-(comm_input-vocab_size).sign(), dim=1)


def get_relevant_state(states, sentence_lengths):
    return torch.gather(states, dim=1, index=(sentence_lengths -1).view(-1, 1, 1).expand(states.size(0), 1, states.size(2)))


class ConvModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        n_filters = 20
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, n_filters, 3, stride=2),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, 3, stride=2),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, 3, stride=2),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, 3, stride=2),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, 3, stride=2),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )

        self.lin = nn.Sequential(
            nn.Linear(180, 50),
            nn.ReLU(),
        )

        self.lin2 = nn.Sequential(
            nn.Linear(50+64, 128),
            nn.ReLU()
        )
        self.lin3 = nn.Linear(128, 2)

        hidden_size = 64

        self.embeddings = nn.Embedding(vocab_size+2, hidden_size, padding_idx=vocab_size)

        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.xavier_uniform_(m.weight)

    def image_rep(self, input):
        batch_size = input.size(0)
        output = self.conv_net(input.transpose(1,3)).view(batch_size, -1)
        output = self.lin(output)

        return output

    def sentence_rep(self, input):
        sentence_lengths = get_sentence_lengths(input, vocab_size=self.vocab_size)

        output = self.embeddings(input)

        output, hidden = self.gru(output)
        output = get_relevant_state(output, sentence_lengths).squeeze(1)

        return output

    def forward(self, input_image, input_text):
        image_rep = self.image_rep(input_image)
        sentence_rep = self.sentence_rep(input_text)

        combined = torch.cat((image_rep, sentence_rep), dim=-1)

        h = self.lin2(combined)
        pred = self.lin3(h)

        return log_softmax(pred, dim=1), softmax(pred, dim=1)[:, 1]
