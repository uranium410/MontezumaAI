import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np

import csv

class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, self.cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output

def mkDataSet(data_size, data_length=200, freq=120., noise=0.1):
    """
    params\n
    data_size : データセットサイズ\n
    data_length : 各データの時系列長\n
    freq : 周波数\n
    noise : ノイズの振幅\n
    returns\n
    train_x : トレーニングデータ（t=1,2,...,size-1の値)\n
    train_t : トレーニングデータのラベル（t=sizeの値）\n
    """
    train_x = []
    train_t = []

    for offset in range(data_size):
        train_x.append([[math.sin(2 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise)] for i in range(data_length)])
        train_t.append([math.sin(2 * math.pi * (offset + data_length) / freq)])

    with open('sin.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(data_length):
            writer.writerow([train_x[0][i][0]])

    with open('sinEnd.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(data_length):
            writer.writerow([train_x[data_length-1][i][0]])

    return train_x, train_t

def mkRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    
    return torch.tensor(batch_x), torch.tensor(batch_t)

def main():
    training_size = 10000
    test_size = 1000
    epochs_num = 1000
    hidden_size = 5
    batch_size = 100

    train_x, train_t = mkDataSet(training_size)
    test_x, test_t = mkDataSet(test_size)

    model = Predictor(1, hidden_size, 1)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    accuracy = []

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0

        celldat = []

        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()

            data, label = mkRandomBatch(train_x, train_t, batch_size)

            output = model(data)
            cell = model.cell
            cell = cell.detach().numpy()
            
            celldat.append(cell)


            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)

        if epoch == 1:
            with open('valStart.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                for i in range(len(celldat)):
                    writer.writerow([celldat[i][0,0,0],
                                 celldat[i][0,0,1],
                                 celldat[i][0,0,2],
                                 celldat[i][0,0,3],
                                 celldat[i][0,0,4]])

        if epoch == 999:
            with open('valEnd.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                for i in range(len(celldat)):
                    writer.writerow([celldat[i][0,0,0],
                                 celldat[i][0,0,1],
                                 celldat[i][0,0,2],
                                 celldat[i][0,0,3],
                                 celldat[i][0,0,4]])

        #test
        test_accuracy = 0.0
        for i in range(int(test_size / batch_size)):
            offset = i * batch_size
            data, label = torch.tensor(test_x[offset:offset+batch_size]), torch.tensor(test_t[offset:offset+batch_size])
            output = model(data, None)

            test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)
        
        training_accuracy /= training_size
        test_accuracy /= test_size

        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))

        accuracy.append(test_accuracy)
    
    with open('accuracy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(accuracy)):
            writer.writerow([accuracy[i]])


if __name__ == '__main__':
    main()