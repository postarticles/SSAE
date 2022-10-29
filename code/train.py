#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from torch.autograd import Variable
import os
import math
from sklearn.model_selection import train_test_split


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cuda:1")
#torch.manual_seed(2022)




# In[3]:


file_path =  r"./4.csv"


# In[4]:


df_data = pd.read_csv(file_path)


# In[5]:


df_data.head(5)


# In[6]:


all_feature_list = df_data.columns
feature_list = all_feature_list[1:-2]
label_list = all_feature_list[-2:]
lable_value = df_data[label_list].values.tolist()

ID_value = df_data[all_feature_list[0]].values 



# In[7]:


class MyDataset(Dataset):
    def __init__(self, raw_data):
     
        self.raw_data = raw_data
    
    def __len__(self):
        return len(self.raw_data)


    def __getitem__(self, idx):

        all_feature_list = self.raw_data.columns
        feature_list = all_feature_list[1:-2]
        label_list = all_feature_list[-2:]
        feature = self.raw_data[feature_list].values[idx]
        label = self.raw_data[label_list].values[idx]
        
        feature = torch.tensor(feature, dtype=torch.float32).to(device)
        label = torch.tensor(label, dtype=torch.float32).to(device)
        return feature, label


# In[8]:


data_dataset = MyDataset(df_data)


# In[9]:


def get_loaders(train_dataset, batch_size, val_ratio, is_shuffle):
	
	dataset_len = int(len(train_dataset))
	test_use_len = int(dataset_len * (1 - val_ratio))
	
	indices = np.random.choice(dataset_len, dataset_len, replace=False)
	# indices = torch.arange(dataset_len)
	train_subset = Subset(train_dataset, indices[:test_use_len])
	test_subset = Subset(train_dataset, indices[test_use_len:])

	data_dataloader = DataLoader(train_dataset, batch_size=batch_size)
	train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=is_shuffle)
	test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=is_shuffle)

	return data_dataloader, train_dataloader, test_dataloader, indices[:test_use_len], indices[test_use_len:]


# In[10]:


data_loder, train_loader, test_loader, train_index, test_index = get_loaders(data_dataset, 128, 0.2, False)
# print(data_loder[0])


# In[11]:


train_label = []
test_label = []
for index in train_index:
	train_label.append(ID_value[index])
for index in test_index:
	test_label.append(ID_value[index])




# In[12]:


class Cox_Regression(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Cox_Regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1, bias=True)
        #self.rul = nn.ReLU()
        #self.fc2 = nn.Linear(hidden_dim, 1)
        ###if gpu is being used
        if torch.cuda.is_available():
            self.fc1 = self.fc1.cuda()
            #self.fc2 = self.fc2.cuda()
        
    def forward(self, x):
        out = self.fc1(x)
        #out = self.rul(out)
        #out = self.fc2(out)
       
        weight = self.fc1.weight.data
        
        weight = weight.clone()
        norm2 = torch.norm(weight, p=2, dim=0)
#         
#         weight = np.linalg.norm(weight,ord=2)
        
        return out, norm2


# In[13]:


class AutoEncoderLayer(nn.Module):

    def __init__(self, input_dim=None, output_dim=None, SelfTraining=False, dropout=None):
        super(AutoEncoderLayer, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.is_training_self = SelfTraining  
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.out_features, bias=True),
            nn.ReLU()  
        self.decoder = nn.Sequential(
            nn.Linear(self.out_features, self.in_features, bias=True),
            nn.ReLU()
        )

        self.regression = nn.Sequential(
            Cox_Regression(self.out_features, self.out_features*2)
        )
        self.dropout = nn.Dropout(p=0.0305)  


    def forward(self, x, y):
        out = self.encoder(x)
        out = self.dropout(out)
        deep_feature = out.clone().detach()
        event_batch = y[:, 0]
        r_batch = np.array([[int(x_i >= y_i) for x_i in y[:, 1]] for y_i in y[:, 1]])
        r_batch = torch.Tensor(r_batch).to(device)
        hazard, weight = self.regression(out)
        if self.is_training_self:
            
            return self.decoder(out), hazard, deep_feature, event_batch, r_batch, weight
        else:
            return out, hazard, deep_feature, event_batch, r_batch, weight


    def lock_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def acquire_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    @property
    def is_training_layer(self):
        return self.is_training_self

    @is_training_layer.setter
    def is_training_layer(self, other: bool):
        self.is_training_self = other


# In[14]:


class StackedAutoEncoder(nn.Module):
    """

    """

    def __init__(self, layers_list=None):
        super(StackedAutoEncoder, self).__init__()
        self.layers_list = nn.ModuleList(layers_list)
        self.layers_num = len(layers_list)
        self.initialize()
    def initialize(self):
        for layer in self.layers_list:
            # assert isinstance(layer, AutoEncoderLayer)
            layer.is_training_layer = False
            # for param in layer.parameters():
            #     param.requires_grad = True

    def forward(self, x, y):
        out = x
        # for layer in self.layers_list:
        #     out = layer(out)
        for index, layer in enumerate(self.layers_list):
            if index == int((self.layers_num / 2) - 1):
                out, hazard1, deep_feature1, event_batch1, r_batch1, weight = layer(out, y)
            else:
                out, hazard, deep_feature, event_batch, r_batch, weight = layer(out, y)
        
        return out, hazard1, deep_feature1, event_batch1, r_batch1, weight




# In[15]:


def get_loss(input_x, hazard, event_batch, r_batch, *args):
    """loss function for all the models"""
    # risk set
    # hazard = normalization(hazard)
    risk_set = torch.matmul(r_batch, torch.exp(hazard))
    risk_set = torch.log(risk_set)
    pl_loss = -torch.sum((hazard - risk_set) * event_batch)/ torch.sum(event_batch)
    # for cox_ae model
    if len(args) == 3:
        # print("args", args)
        output = args[0]  # output of autoencoder
        weight = args[1]
        coefficient = args[2]  # weight of pl_loss
        temp = torch.pow((output - input_x), 2)
        ae_loss = torch.mean(temp)
        all_loss = coefficient* ae_loss + (1-coefficient)*pl_loss
        return pl_loss, ae_loss, all_loss

    return pl_loss


# In[16]:


def train_layers(layers_list=None, layer=None, epoch=None, validate=True, train_loader=None, val_loader=None):
    """
 
    :param layers_list:
    :param layer:
    :param epoch:
    :return:
    """
    if torch.cuda.is_available():
        for model in layers_list:
            model.cuda()
    train_loader = train_loader
    optimizer = torch.optim.Adam(layers_list[layer].parameters(), lr=0.000317)

#     criterion = loss_func()

    # train
    for epoch_index in range(epoch):
        sum_loss = 0.

       
        if layer != 0:
            for index in range(layer):
                layers_list[index].lock_grad()
                layers_list[index].is_training_layer = False  

        for batch_index, (train_data, label) in enumerate(train_loader):
          
            if torch.cuda.is_available():
                train_data = train_data.cuda()  
            train_data = train_data.view(train_data.size(0), -1)
            out = train_data.clone().detach()

           
            if layer != 0:
                for l in range(layer):
                    out, hazard, deep_feature, event_batch, r_batch, weight = layers_list[l](out, label)
            forward_out = out.clone().detach()
                    
         
            out, hazard, deep_feature, event_batch, r_batch, weight = layers_list[layer](out, label)

            optimizer.zero_grad()
            pl_loss, ae_loss, all_loss = get_loss(forward_out, hazard, event_batch, r_batch, out, weight, 0.9617)
            sum_loss += all_loss
            all_loss.backward()
            optimizer.step()
            if (batch_index + 1) % 1 == 0:
                print("Train Layer: {}, Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}".format(
                    layer, (epoch_index + 1), epoch, (batch_index + 1), len(train_loader), all_loss
                ))

        if validate:
            pass


# In[17]:


def train_whole(model=None, epoch=None, validate=True, train_loader=None, val_loader=None,l2_weight=None):
    print(">> start training whole model")
    if torch.cuda.is_available():
        model.cuda()
        
 
    for param in model.parameters():
        param.require_grad = True
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000317)
    #for i in range(epoch):
            #out, hazard, deep_feature, event_batch, r_batch, weight = model(x, label)
            #pl_loss, ae_loss, loss = get_loss(x_train, hazard, event_batch, r_batch, out, weight, 0.01)
            
            #l2_lambda = l2_weight
            #l2_reg = torch.tensor(0.).to(device)
            #for param in model.parameters():
                #l2_reg += torch.norm(param).to(device)
            
            #loss += l2_lambda * l2_reg
            #loss.backward()
            #optimizer.step()
    #model.eval()

    train_loader, val_loader = train_loader, val_loader
    
    best_loss, early_stop_count = math.inf, 0
    
    
    for epoch_index in range(epoch):
#         sum_loss = 0.
        model.train() # Set your model to train mode.
        loss_record = []
        end_loss_record = []
        for batch_index, (train_data, label) in enumerate(train_loader):
            
         
            
            if torch.cuda.is_available():
                train_data = train_data.cuda()
            x = train_data.view(train_data.size(0), -1)
            
            out, hazard, deep_feature, event_batch, r_batch, weight = model(x, label)
            

            pl_loss, ae_loss, loss = get_loss(x, hazard, event_batch, r_batch, out, weight,0.9617)
            
            l2_lambda = l2_weight
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param).to(device)
           
            loss += l2_lambda * l2_reg
#             loss = cu(out, x)
#             sum_loss += loss
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())
            end_loss_record.append(pl_loss.item())

#             if (batch_index + 1) % 10 == 0:
#                 print("Train Whole, Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}".format(
#                     (epoch_index + 1), epoch, (batch_index + 1), len(train_loader), loss
#                 ))

        mean_train_loss = sum(loss_record)/len(loss_record)
        mean_end_loss = sum(end_loss_record)/len(end_loss_record)
        
      
        model.eval() # Set your model to evaluation mode.
        loss_record = []
        end_loss_record = []
        if validate:
            for test_data, labels in test_loader:
                if torch.cuda.is_available():
                    test_data = test_data.cuda()
                    

                
                
                with torch.no_grad():
                    out, hazard, deep_feature, event_batch, r_batch, weight = model(x, labels)
                   
                    pl_loss, ae_loss, loss = get_loss(x, hazard, event_batch, r_batch, out, weight, 0.9617)
#                     loss = cu(out, x)
                loss_record.append(loss.item())
                end_loss_record.append(pl_loss.item())
        mean_valid_loss = sum(loss_record)/len(loss_record)
        mean_valid_end_loss = sum(end_loss_record)/len(end_loss_record)
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model, r"./model（4.14).ckpt")
            print(f'Epoch [{epoch_index+1}/{epoch}]','Saving model with train loss {:.6f}, valid loss {:.6f}, pre loss {:.6f}, pre valid loss {:.6f}'.format(mean_train_loss,best_loss, mean_end_loss, mean_valid_end_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1
#                     print("Test Epoch: {}/{}, Iter: {}/{}, test Loss: {}".format(
#                     epoch_index + 1, epoch, (epoch_index + 1), len(val_loader), loss
#                 ))
    print("<< end training whole model")



# In[18]:


num_tranin_layer_epochs = 30
num_tranin_whole_epochs = 100


# In[19]:


nun_layers = 3
encoder_1 = AutoEncoderLayer(35002, 256,  SelfTraining=True)
#encoder_2 = AutoEncoderLayer(256, 128,  SelfTraining=True)
#decoder_3 = AutoEncoderLayer(128, 256,  SelfTraining=True)
decoder_4 = AutoEncoderLayer(256, 35002,  SelfTraining=True)
layers_list = [encoder_1, decoder_4]


# In[20]:



for level in range(nun_layers - 1):
    train_layers(layers_list=layers_list, layer=level, epoch=num_tranin_layer_epochs, validate=True, train_loader=train_loader)


# In[21]:



SAE_model = StackedAutoEncoder(layers_list=layers_list)


# In[22]:


SAE_model.to(device)


# In[23]:


train_whole(model=SAE_model, epoch=num_tranin_whole_epochs, validate=True, train_loader=train_loader, l2_weight=0.7533)


# In[24]:


def CIndex(model, x_test, label):
    concord = 0.
    total = 0.
    N_test = label[:,0].shape[0]
    out, theta, deep_feature, event_batch, r_batch, weight = model(x_test, label)
    for i in range(N_test):
        if int(label[:,0][i]) == 1:
            for j in range(N_test):
                if label[:,1][j] > label[:,1][i]:
                    total = total + 1
                    if theta[j] < theta[i]: concord = concord + 1
                    elif theta[j] < theta[i]: concord = concord + 0.5

    return(concord/total)


# In[25]:


cindex_list = []
for train_data, label in train_loader:
    cindex_list.append(CIndex(SAE_model, train_data, label))
mean_cindex = sum(cindex_list)/len(cindex_list)


# In[26]:


mean_cindex


# In[27]:


cindex_list = []
for train_data, label in test_loader:
    cindex_list.append(CIndex(SAE_model, train_data, label))
mean_cindex = sum(cindex_list)/len(cindex_list)


# In[28]:


mean_cindex


# In[29]:


def get_all_value(data_loader, SAE_model, name):

	# test_loader = DataLoader(dataset=data_loader, batch_size=len(data_loader), shuffle=False)
	test_loader = data_loader
	status_list = torch.Tensor().to(device)
	time_list = torch.Tensor().to(device)
	pi_list = torch.Tensor().to(device)
	deep_feature_all = torch.Tensor().to(device)
	#cindex_list = []
	if name == 'train':
		ID_all = train_label
	elif name =='test':
		ID_all = test_label
	else:
		return 0

	for test_data, labels in test_loader:
		if torch.cuda.is_available():
			test_data = test_data.cuda()
		x = test_data.view(test_data.size(0), -1)
		with torch.no_grad():
			out, hazard, deep_feature, event_batch, r_batch, weight = SAE_model(x, labels)
			status_list = torch.cat((status_list, labels[:, 0]), dim=0)
			time_list = torch.cat((time_list, labels[:, 1]), dim=0)
			pi_list = torch.cat((pi_list, hazard), dim=0)
			#cindex_list.append(CIndex(SAE_model, test_data, labels))

			deep_feature_all = torch.cat((deep_feature_all, deep_feature), dim=0)
	#mean_cindex = sum(cindex_list) / len(cindex_list)
	#with open("mean_cindex_"+name+str(ind), "a", encoding='utf-8') as f: 
		#print("mean_cindex:", mean_cindex, file=f)
		#f.write('\n')
	#f.close()

	status_list = status_list.cpu().numpy()
	time_list = time_list.cpu().numpy()
	pi_list = pi_list.cpu().numpy().squeeze()
	df_out = pd.DataFrame({
		'ID': ID_all,
		'status': status_list,
		'time': time_list,
		'PI': pi_list
	})
	deep_feature_all = deep_feature_all.cpu().numpy().tolist()

	df_out.to_csv(r"./PI（4.14).csv"+name+'.csv')


	data1 = pd.DataFrame(deep_feature_all)


	data1.to_csv(r"./（4.14).csv"+name+".csv")
	data = pd.read_csv(r"./（4.14).csv"+name+".csv")
	data['ID'] = ID_all
	data.to_csv(r"./（4.14).csv"+name+".csv", mode='a', index=False)


# In[30]:


get_all_value(train_loader, SAE_model,"train")
get_all_value(test_loader, SAE_model,"test")


# In[ ]:




