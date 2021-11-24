import torch
from torchinfo import summary
import argparse
from pydoc import locate
import os
import fnmatch

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


"""
#set the parser args



#load the datasets



#set batch size
batch_size = 128


#performance settings
num_workers = 4
pin_memory = True

#create dataloaders
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

#create the model


#get the device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
model.cuda() 

#print model parameters
#summary(your_model, input_size=(channels, H, W))
summary(model)

#loss functions and optimizer
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
weight_decay = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


#the training loop
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------")
print("Done!")
"""
