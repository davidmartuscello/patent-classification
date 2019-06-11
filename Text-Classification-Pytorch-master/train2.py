import argparse
from main import run_best_model

print('ok')
parser = argparse.ArgumentParser(description='Trains model on the given dataset')
parser.add_argument('-e','--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('-l','--lr', type=float, default=0.00037, help='learning rate')
parser.add_argument('-b','--batch_size', type=int, default=25, help='batch size')
parser.add_argument('-hi','--hidden_size', type=int, default=275, help='hidden size')
parser.add_argument('-c','--cache_data', type=bool, default=False, help='load data into cache (True) or use previously cached data (False)')

print('ok')

args = parser.parse_args()
print('ok')

run_best_model(args)
