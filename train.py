import argparse
import os
import torch
from torch.nn import NLLLoss
import matplotlib.pyplot as plt

import obverter
from data import load_images_dict, get_batches
from model import ConvModel

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Emerging Language')

parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=50, help='number of images in a batch')
parser.add_argument('--num_rounds', type=int, default=20000, help='number of total training rounds')
parser.add_argument('--num_games_per_round', type=int, default=20, help='number of games per round')
parser.add_argument('--vocab_size', type=int, default=5, help='vocabulary size')
parser.add_argument('--max_sentence_len', type=int, default=20, help='maximum sentence length')
parser.add_argument('--data_n_samples', type=int, default=100, help='number of samples per color, shape combination')

args = parser.parse_args()

images_dict = load_images_dict(args.data_n_samples)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent1 = ConvModel(vocab_size=args.vocab_size).to(device)
agent2 = ConvModel(vocab_size=args.vocab_size).to(device)


loss_fn = NLLLoss()

optimizer1 = torch.optim.Adam([p for p in agent1.parameters() if p.requires_grad], args.lr)
optimizer2 = torch.optim.Adam([p for p in agent2.parameters() if p.requires_grad], args.lr)


def get_message(s):
    return ''.join([chr(97+int(v.cpu().data)) for v in s if v < args.vocab_size])


def train_game(speaker, listener, batches, optimizer, max_sentence_len, vocab_size):
    speaker.train(False)
    listener.train(True)

    game_total = 0
    game_correct = 0
    game_loss = 0
    game_sentence_length = 0

    for batch in batches:
        input1, input2, labels, descriptions = batch

        input1, input2 = torch.tensor(input1).float().to(device), torch.tensor(input2).float().to(device)
        labels = torch.tensor(labels).to(device)

        speaker_actions, speaker_probs = obverter.decode(speaker, input1, max_sentence_len, vocab_size, device)

        lg, probs = listener(input2, speaker_actions)
        predictions = torch.round(probs).long()
        correct_vector = (predictions == labels).float()
        n_correct = correct_vector.sum()

        listener_loss = loss_fn(lg, labels.long())

        optimizer.zero_grad()
        listener_loss.backward()
        optimizer.step()

        for t in zip(speaker_actions, speaker_probs, descriptions, labels, probs):
            speaker_action, speaker_prob, description, label, listener_prob = t
            speaker_object, listener_object = description
            message = get_message(speaker_action)
            print("message: '%s', speaker object: %s, speaker score: %.2f, listener object: %s, label: %d, listener score: %.2f" %
                  (message, speaker_object, speaker_prob, listener_object, label.item(), listener_prob.item()))

        print("accuracy", n_correct.item() / len(input1))
        print("listener_loss", listener_loss.item())

        game_correct += n_correct
        game_total += len(input1)
        game_loss += listener_loss * len(input1)
        game_sentence_length += (speaker_actions < vocab_size).sum(dim=1).float().mean() * len(input1)

    game_accuracy = (game_correct / game_total).item()
    game_loss = (game_loss / game_total).item()
    game_sentence_length = (game_sentence_length / game_total).item()

    return game_accuracy, game_loss, game_sentence_length


agent1_accuracy_history = []
agent1_message_length_history = []
agent1_loss_history = []

os.makedirs('checkpoints', exist_ok=True)
for round in range(args.num_rounds):
    print("********** round %d **********" % round)
    batches = get_batches(images_dict, args.data_n_samples, args.num_games_per_round, args.batch_size)

    game_accuracy, game_loss, game_sentence_length = train_game(agent1, agent2, batches, optimizer2, args.max_sentence_len, args.vocab_size)
    print("Game accuracy: %.2f" % (game_accuracy * 100))
    print("Average sentence length: %.1f" % game_sentence_length)
    print("Loss: %.1f" % game_loss)

    agent1_accuracy_history.append(game_accuracy)
    agent1_message_length_history.append(game_sentence_length / 20)
    agent1_loss_history.append(game_loss)

    round += 1
    print("replacing roles")
    print("********** round %d **********" % round)

    game_accuracy, game_loss, game_sentence_length = train_game(agent2, agent1, batches, optimizer1, args.max_sentence_len, args.vocab_size)
    print("Game accuracy: %.2f" % (game_accuracy * 100))
    print("Average sentence length: %.1f" % game_sentence_length)
    print("Loss: %.1f" % game_loss)

    if round % 50 == 0:
        t = list(range(len(agent1_accuracy_history)))
        plt.plot(t, agent1_accuracy_history, label="Accuracy")
        plt.plot(t, agent1_message_length_history, label="Message length (/20)")
        plt.plot(t, agent1_loss_history, label="Training loss")

        plt.xlabel('# Game')
        plt.legend()
        plt.savefig("graph.png")
        plt.clf()

    if round % 500 == 0:
        torch.save(agent1.state_dict(), os.path.join('checkpoints', 'agent1-%d.ckp' % round))
        torch.save(agent2.state_dict(), os.path.join('checkpoints', 'agent2-%d.ckp' % round))