
import random
import torch
import string



data=open("dinos.txt","r").read()

data=data.lower()

chars=list(set(data))

data_size,vocab_size=len(data),len(chars)

#print(f"共有%d个字符，唯一字符有%d"%(data_size,vocab_size))

#char_to_index={ch:i for i ,ch in enumerate(sorted(chars))}
#index_to_char={i:ch for i,ch in enumerate(sorted(chars))}

#print(char_to_index)
all_letters=string.ascii_letters+".,;'-"


def randomChoice(l):
    return l[random.randint(0,len(l)-1)]


def RandomChooseName(name):
    name_choice=randomChoice(name)
    return name_choice

def inputTensor(name):
    tensor=torch.zeros(len(name),1,vocab_size)
    for na in range(len(name)):
        letter=name[na]
        tensor[na][0][all_letters.find(letter)]=1
    return tensor

def targetTensor(name):
    letter_index=[all_letters.find(name[na]) for na in range(1,len(name))]
    letter_index.append(vocab_size-1)
    return torch.LongTensor(letter_index)

def RandomTrainingExample(name):
    name_choice=RandomChooseName(name)
    input_name_tensor=inputTensor(name_choice)
    target_name_tensor=targetTensor(name_choice)
    return input_name_tensor,target_name_tensor