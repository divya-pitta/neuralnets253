
# coding: utf-8

# In[ ]:


import torch
import sys
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt

from torch import nn
from torch.autograd import Variable
from torch import optim
get_ipython().magic(u'matplotlib inline')


# In[ ]:


GPU = torch.cuda.is_available()


# ### Read Data

# In[ ]:


data = open('data/input.txt', 'r').read()

chars = list(set(data))
vocab_size = len(chars)
data, val_data = data[:len(data)*4/5], data[len(data)*4/5:]
data += val_data[:val_data.find("<start>")]
val_data = val_data[val_data.find("<start>"):]

char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}


# ### Model

# In[ ]:


# hyperparameters
# usage: python lstm.py {algo}_{optim}_{num_layer}_{seq_length}_{hidden_size}_{total_epoch}
# model_param = sys.argv[1].split("_")
model_param = "everytune_adam_1_30_150_3".split("_")
hidden_size = int(model_param[4]) # size of hidden layer of neurons
seq_length = int(model_param[3]) # number of steps to unroll the LSTM for
num_layers = int(model_param[2])
model_name = "_".join(model_param)
total_epochs = int(model_param[5])
learning_rate = 1e-4


# In[ ]:


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.hidden2out = nn.Linear(hidden_size, input_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0, c0 = (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)), 
                  Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))
        if GPU:
            h0, c0 = (h0.cuda(), c0.cuda())
        return (h0, c0)

    def forward(self, input_):
        lstm_out, self.hidden = self.lstm(input_, self.hidden)
        output_space = self.hidden2out(lstm_out.view(len(input_), -1))
        output_scores = F.log_softmax(output_space)
        return output_scores


# In[ ]:


model = LSTM(vocab_size, hidden_size, num_layers, 1)
if GPU:
    model = model.cuda()


# In[ ]:


loss_function = nn.NLLLoss()
if "adam" == model_param[1]:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif "rms" == model_param[1]:
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
elif "adagrad" == model_param[1]:
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)


# In[ ]:


log = open("log_{0}.txt".format(model_name), "a")
out = open("tunes_{0}.txt".format(model_name), "a")


# ### Generation

# In[ ]:


def oneHotEncode(char, indexVect):
    temp = [0]*len(indexVect)
    temp[indexVect[char]] = 1
    return temp

def getCharTensor(c):
    temp = oneHotEncode(c, char_to_idx)
    temp = Variable(torch.LongTensor(temp)).cuda() if GPU else Variable(torch.LongTensor(temp))
    temp = temp.view(1,1,-1)
    return temp
    
def getCharLabel(c):
    label = Variable(torch.LongTensor([char_to_idx[c]]))
    return label.cuda() if GPU else label
    
    
def generate(prime_str='<start>', predict_len=10000, temperature=0.4):
    hidden = model.init_hidden()
    prime_input = [getCharTensor(i).float().cuda() if GPU else getCharTensor(i).float() for i in prime_str]
    predicted = prime_str

    for p in range(len(prime_str) - 1):
        _ = model(prime_input[p])
    inp = prime_input[-1]
    
    while True:
        output = model(inp)
        
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        predicted_char = idx_to_char[top_i]
        predicted += predicted_char
        inp = getCharTensor(predicted_char).float().cuda() if GPU else getCharTensor(predicted_char).float()
        
        if predicted[-5:] == '<end>' or len(predicted) >= predict_len:
            break
            
    return predicted


# In[ ]:


## Some preprocessing to split data in tunes for traing and validation
tunes = data.split("<end>\r\n")[:-1]
tunes = [tune + "<end>\r\n" for tune in tunes]

val_tunes = val_data.split("<end>\r\n")
val_tunes = [tune + "<end>\r\n" for tune in val_tunes]


# In[ ]:


def train_random(epochs):
    ran = 1000
    running_loss = 0.0
    val_running_loss = 0.0
    count = 0
    val_count = 0
    
    train_losses = list()
    valid_losses = list()
    
    for epoch in range(epochs):
        rand = np.random.choice(len(data)-seq_length-1, ran)
        for p in rand:
            inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
            targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]
            model.zero_grad()
            model.hidden = model.init_hidden()

            xs = {}
            for t in xrange(len(inputs)):
                x = np.zeros((vocab_size, 1))
                x[inputs[t]] = 1
                xs[t] = Variable(torch.from_numpy(x)).cuda() if GPU else Variable(torch.from_numpy(x))

            xs = torch.cat(xs.values()).float().view(len(xs.values()), 1, -1)
            xs = xs.cuda() if GPU else xs
            ys = Variable(torch.LongTensor(targets)).cuda() if GPU else Variable(torch.LongTensor(targets))

            pred = model(xs)

            loss = loss_function(pred, ys)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data.cpu().numpy()[0]
            count += 1
            
        rand = np.random.choice(len(val_data)-seq_length-1, ran/5)
        for p in rand:
            inputs = [char_to_idx[ch] for ch in val_data[p:p+seq_length]]
            targets = [char_to_idx[ch] for ch in val_data[p+1:p+seq_length+1]]
            model.hidden = model.init_hidden()

            xs = {}
            for t in xrange(len(inputs)):
                x = np.zeros((vocab_size, 1))
                x[inputs[t]] = 1
                xs[t] = Variable(torch.from_numpy(x)).cuda() if GPU else Variable(torch.from_numpy(x))

            xs = torch.cat(xs.values()).float().view(len(xs.values()), 1, -1)
            xs = xs.cuda() if GPU else xs
            ys = Variable(torch.LongTensor(targets)).cuda() if GPU else Variable(torch.LongTensor(targets))

            pred = model(xs)

            loss = loss_function(pred, ys)
            
            val_running_loss += loss.data.cpu().numpy()[0]
            val_count += 1
            
        if  (running_loss *1.0 / count) < 2.5:
            tune = generate()
            if tune[-5:] == "<end>":
                out.write(tune)
                out.write("\n======\n")
                out.flush()
            
        log.write("[{0}]: RLoss: {1}, Val_RLoss: {2}\n".format(epoch, running_loss*1.0/count, val_running_loss*1.0/val_count))
        log.flush()
            
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'trained_model_{0}.pt'.format(model_name))
            
        
        train_losses.append(running_loss * 1.0 / count)
        valid_losses.append(val_running_loss*1.0/val_count)
            
        if epoch > 0 and valid_losses[epoch] >= valid_losses[epoch-1]: 
            #print "Validation Loss Increased. Decreasing Patience."
            log.write("Validation Loss Increased. Decreasing Patience.\n")
            log.flush()
            patience -= 1 
        else: 
            patience = 3 
            
        if patience <= 0 or epoch == (epochs - 1):
            #print "Validation loss increased consecutively for 3 epochs. Stopping learning."
            log.write("Validation loss increased consecutively for 3 epochs. Stopping learning.\n")
            log.flush()
            break
        
        pickle_out = open('output_{0}'.format(model_name), "w")
        pickle.dump([train_losses, valid_losses], pickle_out)
        pickle_out.close()


# In[ ]:


def train_special(epochs):
    running_loss = 0.0
    val_running_loss = 0.0
    count = 0
    val_count = 0
    
    train_losses = list()
    valid_losses = list()
    
    for epoch in range(epochs):
        selected_tunes = np.random.randint(0, len(tunes), 10)
        for idx in selected_tunes:
            for p in range(0, len(tunes[idx]) - seq_length - 1, seq_length):
                inputs = [char_to_idx[ch] for ch in tunes[idx][p:p+seq_length]]
                targets = [char_to_idx[ch] for ch in tunes[idx][p+1:p+seq_length+1]]
                model.zero_grad()
                model.hidden = model.init_hidden()

                xs = {}
                for t in xrange(len(inputs)):
                    x = np.zeros((vocab_size, 1))
                    x[inputs[t]] = 1
                    xs[t] = Variable(torch.from_numpy(x)).cuda() if GPU else Variable(torch.from_numpy(x))

                xs = torch.cat(xs.values()).float().view(len(xs.values()), 1, -1)
                xs = xs.cuda() if GPU else xs
                ys = Variable(torch.LongTensor(targets)).cuda() if GPU else Variable(torch.LongTensor(targets))

                pred = model(xs)

                loss = loss_function(pred, ys)
                loss.backward()
                optimizer.step()

                running_loss += loss.data.cpu().numpy()[0]
                count += 1
            
            val_selected_tunes = np.random.randint(0, len(val_tunes), 3)
            for idx in val_selected_tunes:
                for p in range(0, len(val_tunes[idx]) - seq_length - 1, seq_length):
                    inputs = [char_to_idx[ch] for ch in val_data[p:p+seq_length]]
                    targets = [char_to_idx[ch] for ch in val_data[p+1:p+seq_length+1]]
                    model.hidden = model.init_hidden()

                    xs = {}
                    for t in xrange(len(inputs)):
                        x = np.zeros((vocab_size, 1))
                        x[inputs[t]] = 1
                        xs[t] = Variable(torch.from_numpy(x)).cuda() if GPU else Variable(torch.from_numpy(x))

                    xs = torch.cat(xs.values()).float().view(len(xs.values()), 1, -1)
                    xs = xs.cuda() if GPU else xs
                    ys = Variable(torch.LongTensor(targets)).cuda() if GPU else Variable(torch.LongTensor(targets))

                    pred = model(xs)

                    loss = loss_function(pred, ys)

                    val_running_loss += loss.data.cpu().numpy()[0]
                    val_count += 1
            
        if  (running_loss *1.0 / count) < 2.5:
            tune = generate()
            if tune[-5:] == "<end>":
                out.write(tune)
                out.write("\n======\n")
                out.flush()
            
        log.write("[{0}]: RLoss: {1}, Val_RLoss: {2}\n".format(epoch, running_loss*1.0/count, val_running_loss*1.0/val_count))
        log.flush()
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'trained_model_{0}.pt'.format(model_name))
            
        train_losses.append(running_loss * 1.0 / count)
        valid_losses.append(val_running_loss*1.0/val_count)
            
        if epoch > 0 and valid_losses[epoch] >= valid_losses[epoch-1]: 
            #print "Validation Loss Increased. Decreasing Patience."
            log.write("Validation Loss Increased. Decreasing Patience.\n")
            log.flush()
            patience -= 1 
        else: 
            patience = 3 
            
        if patience <= 0 or epoch == (epochs - 1):
            #print "Validation loss increased consecutively for 3 epochs. Stopping learning."
            log.write("Validation loss increased consecutively for 3 epochs. Stopping learning.\n")
            log.flush()
            pickle_out = open('output_{0}'.format(model_name), "w")
            pickle.dump([train_losses, valid_losses], pickle_out)
            pickle_out.close()
            break
            

#         print "target======>", data[p+1:p+seq_length+1]
#         print "output======>", "".join([idx_to_char[c] for c in list(np.argmax(pred.data.cpu().numpy(), axis=1))])


# In[ ]:


def train_every_tune(epochs):
    running_loss = 0.0
    val_running_loss = 0.0
    count = 0
    val_count = 0

    train_losses = list()
    valid_losses = list()
    
    for epoch in range(epochs):
        for idx in range(0,len(tunes)):
            p = np.random.randint(len(tunes[idx]) - seq_length - 1)
            inputs = [char_to_idx[ch] for ch in tunes[idx][p:p+seq_length]]
            targets = [char_to_idx[ch] for ch in tunes[idx][p+1:p+seq_length+1]]
            model.zero_grad()
            model.hidden = model.init_hidden()

            xs = {}
            for t in xrange(len(inputs)):
                x = np.zeros((vocab_size, 1))
                x[inputs[t]] = 1
                xs[t] = Variable(torch.from_numpy(x)).cuda() if GPU else Variable(torch.from_numpy(x))

            xs = torch.cat(xs.values()).float().view(len(xs.values()), 1, -1)
            xs = xs.cuda() if GPU else xs
            ys = Variable(torch.LongTensor(targets)).cuda() if GPU else Variable(torch.LongTensor(targets))

            pred = model(xs)

            loss = loss_function(pred, ys)
            loss.backward()
            optimizer.step()

            running_loss += loss.data.cpu().numpy()[0]
            count += 1
            
        
        for idx in range(0, len(val_tunes)):
            p = np.random.randint(len(val_tunes[idx]) - seq_length - 1)
            inputs = [char_to_idx[ch] for ch in val_tunes[idx][p:p+seq_length]]
            targets = [char_to_idx[ch] for ch in val_tunes[idx][p+1:p+seq_length+1]]
            model.hidden = model.init_hidden()

            xs = {}
            for t in xrange(len(inputs)):
                x = np.zeros((vocab_size, 1))
                x[inputs[t]] = 1
                xs[t] = Variable(torch.from_numpy(x)).cuda() if GPU else Variable(torch.from_numpy(x))

            xs = torch.cat(xs.values()).float().view(len(xs.values()), 1, -1)
            xs = xs.cuda() if GPU else xs
            ys = Variable(torch.LongTensor(targets)).cuda() if GPU else Variable(torch.LongTensor(targets))

            pred = model(xs)

            loss = loss_function(pred, ys)

            val_running_loss += loss.data.cpu().numpy()[0]
            val_count += 1
            
        if  (running_loss *1.0 / count) < 2.5:
            tune = generate()
            if tune[-5:] == "<end>":
                out.write(tune)
                out.write("\n======\n")
                out.flush()
            
        log.write("[{0}]: RLoss: {1}, Val_RLoss: {2}\n".format(epoch, running_loss*1.0/count, val_running_loss*1.0/val_count))
        log.flush()
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'trained_model_{0}.pt'.format(model_name))
            
        train_losses.append(running_loss * 1.0 / count)
        valid_losses.append(val_running_loss*1.0/val_count)
            
        if epoch > 0 and valid_losses[epoch] >= valid_losses[epoch-1]: 
            #print "Validation Loss Increased. Decreasing Patience."
            log.write("Validation Loss Increased. Decreasing Patience.\n")
            log.flush()
            patience -= 1 
        else: 
            patience = 3 
            
        if patience <= 0 or epoch == (epochs - 1):
            #print "Validation loss increased consecutively for 3 epochs. Stopping learning."
            log.write("Validation loss increased consecutively for 3 epochs. Stopping learning.\n")
            log.flush()
            pickle_out = open('output_{0}'.format(model_name), "w")
            pickle.dump([train_losses, valid_losses], pickle_out)
            pickle_out.close()
            break
            

#         print "target======>", data[p+1:p+seq_length+1]
#         print "output======>", "".join([idx_to_c


# In[ ]:


def train_every_tune_curriculum(epochs, early_stopping=False):
    running_loss = 0.0
    val_running_loss = 0.0
    count = 0
    val_count = 0

    train_losses = list()
    valid_losses = list()
    seq_length = 5
    for epoch in range(epochs):
        if(epoch%100==0 and seq_length*2 <= 100):
            seq_length = seq_length * 2
            log.write("Changing seq length to {0}".format(seq_length))
            log.flush()
        for idx in range(0,len(tunes)):
            tuneLen = len(tunes[idx])
            if(tuneLen > seq_length):
                p = np.random.randint(tuneLen - seq_length - 1)
                inputs = [char_to_idx[ch] for ch in tunes[idx][p:p+seq_length]]
                targets = [char_to_idx[ch] for ch in tunes[idx][p+1:p+seq_length+1]]
            else:
                inputs = [char_to_idx[ch] for ch in tunes[idx][0:tuneLen-1]]
                targets = [char_to_idx[ch] for ch in tunes[idx][1:tuneLen]]
            model.zero_grad()
            model.hidden = model.init_hidden()

            xs = {}
            for t in range(len(inputs)):
                x = np.zeros((vocab_size, 1))
                x[inputs[t]] = 1
                xs[t] = Variable(torch.from_numpy(x)).cuda() if GPU else Variable(torch.from_numpy(x))

            xs = torch.cat(list(xs.values())).float().view(len(xs.values()), 1, -1)
            xs = xs.cuda() if GPU else xs
            ys = Variable(torch.LongTensor(targets)).cuda() if GPU else Variable(torch.LongTensor(targets))

            pred = model(xs)

            loss = loss_function(pred, ys)
            loss.backward()
            optimizer.step()

            running_loss += loss.data.cpu().numpy()[0]
            count += 1
            
        
        for idx in range(0, len(val_tunes)):
            tuneLen = len(val_tunes[idx])
            if(tuneLen > seq_length):
                p = np.random.randint(tuneLen - seq_length - 1)
                inputs = [char_to_idx[ch] for ch in val_tunes[idx][p:p+seq_length]]
                targets = [char_to_idx[ch] for ch in val_tunes[idx][p+1:p+seq_length+1]]
            else:
                inputs = [char_to_idx[ch] for ch in val_tunes[idx][0:tuneLen-1]]
                targets = [char_to_idx[ch] for ch in val_tunes[idx][1:tuneLen]]
            model.hidden = model.init_hidden()

            xs = {}
            for t in range(len(inputs)):
                x = np.zeros((vocab_size, 1))
                x[inputs[t]] = 1
                xs[t] = Variable(torch.from_numpy(x)).cuda() if GPU else Variable(torch.from_numpy(x))

            xs = torch.cat(list(xs.values())).float().view(len(xs.values()), 1, -1)
            xs = xs.cuda() if GPU else xs
            ys = Variable(torch.LongTensor(targets)).cuda() if GPU else Variable(torch.LongTensor(targets))

            pred = model(xs)

            loss = loss_function(pred, ys)

            val_running_loss += loss.data.cpu().numpy()[0]
            val_count += 1
            
        if  (running_loss *1.0 / count) < 2.5:
            tune = generate()
            if tune[-5:] == "<end>":
                out.write(tune)
                out.write("\n======\n")
                out.flush()
            
        log.write("[{0}]: RLoss: {1}, Val_RLoss: {2}\n".format(epoch, running_loss*1.0/count, val_running_loss*1.0/val_count))
        log.flush()
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'trained_model_{0}.pt'.format(model_name))
            
        train_losses.append(running_loss * 1.0 / count)
        valid_losses.append(val_running_loss*1.0/val_count)
            
        if early_stopping and (epoch > 0 and valid_losses[epoch] >= valid_losses[epoch-1]): 
            #print "Validation Loss Increased. Decreasing Patience."
            log.write("Validation Loss Increased. Decreasing Patience.\n")
            log.flush()
            patience -= 1 
        else: 
            patience = 3 
            
        if patience <= 0 or epoch == (epochs - 1):
            #print "Validation loss increased consecutively for 3 epochs. Stopping learning."
            log.write("Validation loss increased consecutively for 3 epochs. Stopping learning.\n")
            log.flush()
#             pickle_out = open('output_{0}'.format(model_name), "w")
#             pickle.dump([train_losses, valid_losses], pickle_out)
#             pickle_out.close()
            pickle.dump([train_losses, valid_losses], open('output_{0}'.format(model_name), 'wb'))
            pickle.dump(train_losses, open('trainloss_{0}'.format(model_name), 'wb'))
            pickle.dump(valid_losses, open('validloss_{0}'.format(model_name), 'wb'))
            break
    return train_losses, valid_losses
#         print "target======>", data[p+1:p+seq_length+1]
#         print "output======>", "".join([idx_to_c


# In[ ]:


if "random" == model_param[0]:
    train_random(total_epochs)
elif "special" == model_param[0]:
    train_special(total_epochs)
elif "everytune" == model_param[0]:
    train_every_tune(total_epochs)
elif "curriculum" == model_param[0]:
    train_every_tune_curriculum(total_epochs)


# In[ ]:


def outputs(outfile):
    with open(outfile,'rb') as f: 
        trl, cvl = pickle.load(f)
    epo = range(len(cvl))

    print("{} & {} & {}".format(len(cvl), trl[cvl.index(min(cvl))], min(cvl)))
    ## Overall Loss
    plt.xlabel("Iterations")
    plt.ylabel("Cross Entropy Loss")
    plt.plot(epo,trl,label='Training')
    plt.plot(epo,cvl,label='Validation')
    plt.title("Loss VS Epochs")
    plt.legend()
    plt.show()


# In[ ]:


outputs('output_{0}'.format(model_name))


# In[ ]:


import pandas as pd

def generate_with_featmodel(model, prime_str='<', predict_len=500, temperature=0.4):
    prime_input = [getCharTensor(i).float().cuda() for i in prime_str]
    predicted = prime_str
    # Use priming string to "build up" hidden state
    
    hidden_state_matrix = np.copy(model.hidden[0].cpu().unsqueeze(0).data.numpy())
    print(hidden_state_matrix.shape)
    hidden_state_matrix = hidden_state_matrix.reshape((1,hidden_state_matrix.size))
    cell_state_matrix = np.copy(model.hidden[1].cpu().unsqueeze(0).data.numpy())
    cell_state_matrix = cell_state_matrix.reshape((1,cell_state_matrix.size))
    hidden_layers = model.hidden[0].size()[0]
    
    for p in range(len(prime_str) - 1):
        model(prime_input[p])
        curr_hidden_state = model.hidden[0][0,0,:].cpu().data.numpy()
        curr_cell_state = model.hidden[1][0,0,:].cpu().data.numpy()
        for h in range(1, hidden_layers):
            curr_hidden_state = np.concatenate((curr_hidden_state, model.hidden[0][h,0,:].cpu().data.numpy()))
            curr_cell_state = np.concatenate((curr_cell_state, model.hidden[1][h,0,:].cpu().data.numpy()))
        hidden_state_matrix = np.vstack((hidden_state_matrix, curr_hidden_state))
        cell_state_matrix = np.vstack((cell_state_matrix, curr_cell_state))
    inp = prime_input[-1]
    
    while(True):
        output = model(inp)
        curr_hidden_state = model.hidden[0][0,0,:].cpu().data.numpy()
        curr_cell_state = model.hidden[1][0,0,:].cpu().data.numpy()
        for h in range(1, hidden_layers):
            curr_hidden_state = np.concatenate((curr_hidden_state, model.hidden[0][h,0,:].cpu().data.numpy()))
            curr_cell_state = np.concatenate((curr_cell_state, model.hidden[1][h,0,:].cpu().data.numpy()))
        hidden_state_matrix = np.vstack((hidden_state_matrix, curr_hidden_state))
        cell_state_matrix = np.vstack((cell_state_matrix, curr_cell_state))
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = idx_to_char[top_i]
        predicted += predicted_char
        inp = getCharTensor(predicted_char).float().cuda()
        
        if(predicted[-5:]=='<end>' or len(predicted)>=predict_len):
            break
            
    hidden_state_matrix = np.delete(hidden_state_matrix, 0, 0)
    cell_state_matrix = np.delete(cell_state_matrix, 0, 0)
    df_hidden_state = pd.DataFrame(hidden_state_matrix)
    df_cell_state = pd.DataFrame(cell_state_matrix)

    df_hidden_file = "df_hidden1.csv".format(model_name)
    df_cell_file = "df_cell1.csv".format(model_name)
    df_hidden_state.to_csv(df_hidden_file)
    df_cell_state.to_csv(df_cell_file)

    return predicted, df_hidden_file, df_cell_file


# In[ ]:


from math import pi
from bokeh.io import show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, BasicTicker, PrintfTickFormatter,ColorBar
from bokeh.models import FuncTickFormatter
from bokeh.plotting import figure
from bokeh.io import output_notebook
from bokeh.layouts import row, column
import os
output_notebook()

predictionSeq = ""

def plot_heatmap(df_file, predictedSeq):
    data = pd.read_csv(df_file, index_col = 0)
    data.index.name = 'chars'
    data.columns.name = 'cell'
    cell = list(data.columns)

    index = {i:predictedSeq[i] for i in range(len(data.index))}

    seq = [str(i) for i in data.index]
    cell = list(data.columns)
    print(cell)

    df = pd.DataFrame(data.stack(), columns=['value']).reset_index()
    print(df.shape)
    colors = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]

    colors.reverse()
    mapper = LinearColorMapper(palette=colors, low=-2, high=2)#low=df.value.min(), high=df.value.max())
    source = ColumnDataSource(df)
    TOOLS = "hover,pan,reset,save,wheel_zoom"

    y_total_range = list(reversed(cell))
    print("Range = ", len(y_total_range))
    p = figure(title="GRU Hidden State Activations",  x_range=seq, y_range=y_total_range[:100], x_axis_location="above", plot_width=3500, plot_height=2000,
               tools=TOOLS, toolbar_location='below')

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3
    p.xaxis.formatter = FuncTickFormatter(code="""
    var labels = %s;
    return labels[tick];
    """%index)

    p.rect(x="chars", y="cell", width=1, height=1, source=source, fill_color={'field': 'value', 'transform': mapper},
           line_color=None)

    p.select_one(HoverTool).tooltips = [('value', '@value')]
    
    p1 = figure(title="GRU Hidden State Activations 2nd hidden layer",  x_range=seq, y_range=y_total_range[100:], x_axis_location="above", plot_width=3500, plot_height=2000,
               tools=TOOLS, toolbar_location='below')

    p1.grid.grid_line_color = None
    p1.axis.axis_line_color = None
    p1.axis.major_tick_line_color = None
    p1.axis.major_label_text_font_size = "8pt"
    p1.axis.major_label_standoff = 0
    p1.xaxis.major_label_orientation = pi / 3
    p1.xaxis.formatter = FuncTickFormatter(code="""
    var labels = %s;
    return labels[tick];
    """%index)

    p1.rect(x="chars", y="cell", width=1, height=1, source=source, fill_color={'field': 'value', 'transform': mapper},
           line_color=None)

    p1.select_one(HoverTool).tooltips = [('value', '@value')]
#     s = vplot(p, p1)
    show(column(p, p1))     # show the plot

def generate_featmodel_heatmap():
    global predictionSeq
    print(model)
    predictedSeq, df_hidden_file, df_cell_file = generate_with_featmodel(model,prime_str="<start>")
    predictionSeq = predictedSeq
    print("Predicted sequence = ", predictionSeq)
    plot_heatmap(df_hidden_file, predictedSeq)
    plot_heatmap(df_cell_file, predictedSeq)
    
generate_featmodel_heatmap()


# In[ ]:


import seaborn as sns

print(predictionSeq)

data = pd.read_csv("df_hidden1.csv", index_col = 0)
data.index.name = 'chars'
data.columns.name = 'cell'
cell = list(data.columns)

predictedSeq = predictionSeq

index = {i:predictedSeq[i] for i in range(len(data.index))}

seq = [str(i) for i in data.index]
cell = list(data.columns)

labels = np.array(list(predictedSeq))

labels = labels[:-10].reshape(49, 10)
ploty = data.iloc[:-9, 96].reshape(49,10)
fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(ploty, annot=labels, fmt='', ax=ax)
plt.show()

