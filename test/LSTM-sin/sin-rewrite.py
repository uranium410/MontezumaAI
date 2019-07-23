import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np
import random
import csv
import sys

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

Episodes = 100
Steps = 200
Data_length=400
freq = 100
noise = 0.1

progressDisp = ['-','\\','|','/']

def main():
    model = Predictor(1,1,1)

    for eps in range(Episodes):
        if eps%10 == 0:
            print('\r Episode:',eps)

        dat = []
        correct = []

        predict = []
        losses = []
        randomStart = random.random()*2*math.pi

        for i in range(Data_length):
            dat.append(math.sin(2*math.pi*i/freq + randomStart + np.random.normal(loc=0.0, scale=noise)))
            correct.append(math.sin(2*math.pi*(i+1)/freq + randomStart))

        for step in range(Steps):
            inputDat = torch.tensor([[[dat[i]]]])

            predict.append(model(inputDat))
            losses.append(nn.MSELoss(predict[step],correct[step]))

        print('\r training.',progressDisp[eps % 4] , end="")



    print('\r done         ')

if __name__ == '__main__':
    main()