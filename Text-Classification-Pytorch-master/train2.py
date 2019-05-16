import argparse
from main import *


parser = argparse.ArgumentParser(description='Trains model on the given dataset')
parser.add_argument('-e','--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('-l','--lr', type=float, default=0.00037, help='learning rate')
parser.add_argument('-b','--batch_size', type=int, default=25, help='batch size')
parser.add_argument('-hi','--hidden_size', type=int, default=275, help='hidden size')

args = parser.parse_args()

run_best_model(args)
