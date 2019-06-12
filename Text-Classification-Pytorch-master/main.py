
import os
import time
import load_patents
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.selfAttention import SelfAttention
from models.LSTM import LSTMClassifier
from models.LSTM_Attn import AttentionModel
from models.RCNN import RCNN
from models.RNN import RNN
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, train_iter, epoch, batch_size, learning_rate):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not batch_size):# One of the batch returned by BucketIterator has length different than batch_size.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = F.cross_entropy(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter, batch_size):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not batch_size):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = F.cross_entropy(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)

def objective(batch_size, hidden_size, learning_rate):
    batch_size = int(batch_size)
    hidden_size = int(hidden_size)
    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_patents.load_dataset(batch_size, cache_data=False)
    output_size = 2
    embedding_length = 300
    weights = word_embeddings
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    #model = AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, weights)
    #model = RNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, weights)
    #model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, weights)
    #model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, weights)
    loss_fn = F.cross_entropy

    for epoch in range(10):
        #(model, train_iter, epoch, batch_size, learning_rate)
        train_loss, train_acc = train_model(model, train_iter, epoch, batch_size, learning_rate)
        val_loss, val_acc = eval_model(model, valid_iter, batch_size)

    test_loss, test_acc = eval_model(model, test_iter, batch_size)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

    return test_acc

def parameters_tuning():
    pbounds = {'batch_size': (16, 64), 'hidden_size': (150, 300), 'learning_rate':(0.0001, 0.01)}
    print("LSTM, 10000, random5, bayes5")
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(
        init_points=5,
        n_iter=5,
    )
    print(optimizer.max)

def run_best_model(args):
    learning_rate = args.lr
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    epochs = args.epochs
    cache_data = args.cache_data
    output_size = 2
    embedding_length = 300
    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_patents.load_dataset(batch_size, cache_data=cache_data)
    weights = word_embeddings
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

    acc_list = []
    val_acc = []
    for epoch in range(epochs):
        #(model, train_iter, epoch, batch_size, learning_rate)
        train_loss, train_acc = train_model(model, train_iter, epoch, batch_size, learning_rate)
        val_loss, val_acc = eval_model(model, valid_iter, batch_size)
        print(f'EPOCH {epoch} -- Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
        acc_list.append(train_acc)
        val_list.append(val_acc)

    plt.plot(acc_list, label="train")
    plt.plot(val_list, label="val")
    plt.savefig('acc_graph.png')

    test_loss, test_acc = eval_model(model, test_iter, batch_size)
    print("performance of model:")
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

#parameters_tuning()
