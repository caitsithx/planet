import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import fbeta_score
import settings
from utils import save_array, load_array
torch.backends.cudnn.enabled = False

RESULT_DIR = settings.RESULT_DIR
PRED_VAL_RAW = RESULT_DIR + '/pred_val_raw.dat'
VAL_LABELS = RESULT_DIR + '/val_labels.dat'

epochs = 100

thr = [0.24, 0.29, 0.17, 0.16, 0.37, 0.26, 0.23, 0.27, 0.19, 0.32, 0.11, 0.1, 0.18, 0.36, 0.27, 0.4, 0.07]

def f_measure(logits, labels, threshold=0.23, beta=2):
    logits = logits.cpu().numpy()
    p = np.zeros_like(logits)
    for i in range(17):
      p[:, i] = (logits[:, i] > thr[i]).astype(np.int)
    score = fbeta_score(labels.cpu().numpy(), p, beta=2, average='samples')
    return score
    
class SimpleNet(nn.Module):
    def __init__(self, num_models):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(num_models, 1, 1, bias=False)
    
    def forward(self, x):
        return F.relu(self.conv1(x))


def train():
    preds = np.expand_dims(np.array(load_array(PRED_VAL_RAW)).astype(np.float32), axis=0)
    labels = np.array(load_array(VAL_LABELS)).astype(np.float32)[0]
    labels = np.expand_dims(np.expand_dims(labels, axis=0), axis=0)

    data = Variable(torch.from_numpy(preds).cuda())
    labels = Variable(torch.from_numpy(labels).cuda())
    print(data.size())
    print(labels.size())

    net = SimpleNet(data.size()[1]).cuda()
    net.train(True)
    
    criterion = nn.L1Loss() #nn.MSELoss(), 
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

    #print(net.conv1.weight)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('epoch:{}'.format(epoch))
        print('loss:{}'.format(loss.data[0]))
        #print(outputs.data[0][0].size())
        score = f_measure(outputs.data[0][0], labels.data[0][0])
        print('score:{}'.format(score))
    
    print(net.conv1.weight)

train()