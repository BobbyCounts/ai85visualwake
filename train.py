import torch
from torchinfo import summary
import argparse
from pydoc import locate
import os
import fnmatch

import torch.nn.functional as F

import datasets.vww.vww as vww

#create the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="set the model to train")
parser.add_argument("--batch_size", help="training batch size")

args = parser.parse_args()



#dynamically load models
for _, _, files in os.walk('models'):
    for name in files:
        if fnmatch.fnmatch(name, '*.py'):
            fn = 'models.' + name[:-3]
            m = locate(fn)
            for i in m.models:
                print(i)

training_data, test_data = vww.vww_get_datasets()

#I need to build the parser later
#Just manually define the model hur
Model = locate('models.wakeModel.wakeModel')
model = Model()



#set the parser args



#load the datasets



#set batch size
batch_size = 128


#performance settings
num_workers = 4
pin_memory = True

#create dataloaders
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

#create the model







#get the device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
model.cuda() 


#print model parameters
#summary(your_model, input_size=(channels, H, W))
summary(model)

#loss functions and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
weight_decay = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

"""
#the training loop
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------")
print("Done!")
"""


def train(model, optimizer, loss_fn, train_dataloader, test_dataloader, epochs=20, device="cpu"):
    for epoch in range(1, epochs+1):
        model.to(device)
        training_loss = 0.0
        val_loss = 0.0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_dataloader.dataset)
        
        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in test_dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            val_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1],
                               targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        val_loss /= len(test_dataloader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        val_loss, num_correct / num_examples))

train(model, optimizer, loss_fn, train_dataloader, test_dataloader, 20, device)      

