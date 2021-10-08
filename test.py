import torch
from model import RNN
import random
from data_preprocessing import inputTensor

m_state_dict=torch.load('rnn.pt')
max_length=10
data=open("dinos.txt","r").read()
data=data.lower()
name=data.split('\n')
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)


new_m=RNN(vocab_size,128,vocab_size)
new_m.load_state_dict(m_state_dict)

def sample(start_letter):
    
    with torch.no_grad(): # no need to track history in sampling
        input=inputTensor(start_letter)
        hidden=new_m.initHidden()

        output_name=start_letter

        for  i in range(max_length):
            output,hidden=new_m(input[0],hidden)
            topv,topi=output.topk(1)
            topi=topi[0][0]
            if chars[topi] == '\n':
                break
            else:
                letter=chars[topi]
                output_name+=letter

            input=inputTensor(letter)

        return output_name
    
letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','\n']

if __name__=="__main__":
    for i in range(200):
        
        ran=random.randint(0,len(letters)-1)
        if letters[ran]=='\n':
            continue
        print(sample(letters[ran]))