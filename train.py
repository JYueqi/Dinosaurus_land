import math
import time

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

import torch.nn as nn

from data_preprocessing import RandomTrainingExample,inputTensor
from model import RNN

data=open("dinos.txt","r").read()
data=data.lower()
name=data.split('\n')
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)

criterion =nn.NLLLoss()

lr=0.0005
rnn=RNN(vocab_size,128,vocab_size)

def train(input_name_tensor,target_name_tensor):
    target_name_tensor.unsqueeze_(-1)
    hidden=rnn.initHidden()

    rnn.zero_grad()

    loss=0

    for i in range(input_name_tensor.size(0)):
        output,hidden=rnn(input_name_tensor[1],hidden)
        l=criterion(output,target_name_tensor[i])
        loss+=l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-lr,p.grad.data)

    return output,loss.item()/input_name_tensor.size(0)


def timeSince(since):
    now=time.time()
    s=now-since
    m=math.floor(s/60)
    s-=m*60
    return "%dm %ds" %(m,s)


max_length=20

#手动选择一个字母作为初始字母，也可以使用random随机选择
#但是这种方式只是由于简单而在练习中使用，在实际应用时，应该对数据集进行处理
#在每个名字前面加一个<SOS>标记并作为一个字符，然后进行训练，这样可以让网络自己选择名字开头的第一个字母

def sample(start_letter='A'):
    
    with torch.no_grad(): # no need to track history in sampling
        input=inputTensor(start_letter)
        hidden=rnn.initHidden()

        output_name=start_letter

        for  i in range(max_length):
            output,hidden=rnn(input[0],hidden)
            topv,topi=output.topk(1)
            topi=topi[0][0]
            if chars[topi] == '\n':
                break
            else:
                letter=chars[topi]
                output_name+=letter

            input=inputTensor(letter)

        return output_name


        





if __name__=="__main__":
    n_iter=100000
    print_every=5000
    plot_every=500
    all_losses=[]
    total_loss=0

    start=time.time()
    
    
    
    
    for iter in range(1,n_iter+1):
        output,loss=train(*RandomTrainingExample(name))
        total_loss+=loss

        if iter%print_every==0:
            print("%s (%d %d%%) %.4f" %(timeSince(start),iter,iter/n_iter*100,loss))

        if iter%plot_every==0:
            all_losses.append(total_loss/plot_every)
            total_loss=0
    
    torch.save(rnn.state_dict(),'rnn.pt')
    
    plt.figure()
    plt.plot(all_losses)
    plt.savefig('./test.jpg')

'''
print(sample('a'))
    print(sample('b'))
    print(sample('s'))
    print(sample('g'))
    print(sample('d'))
    print(sample('e'))
    print(sample('c'))
    
'''
    
