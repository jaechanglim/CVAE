from model import CVAE
from utils import *
import numpy as np
import os
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
latent_size = 200
batch_size = 128
seq_length = 120

molecules, char, labels, length = load_data('smiles_prop.txt', seq_length)

vocab_size = len(char)

num_train_data = int(len(molecules)*0.75)
train_molecules = molecules[0:num_train_data]
test_molecules = molecules[num_train_data:-1]

train_labels = labels[0:num_train_data]
test_labels = labels[num_train_data:-1]

train_length = length[0:num_train_data]
test_length = length[num_train_data:-1]

model = CVAE(vocab_size,
             batch_size = batch_size,
             latent_size = latent_size,
             stddev = 1.0,
             mean = 0.0,
             )

num_epochs    = 200
save_every    = 500
learning_rate = 0.0001
temperature = 1.0
min_temperature = 0.5
decay_rate    = 0.95
for epoch in range(num_epochs):
    # Learning rate scheduling 
    st = time.time()
    model.assign_lr(learning_rate * (decay_rate ** epoch))
    train_loss = []
    test_loss = []
    st = time.time()
    
    for iteration in range(int(len(train_molecules)/batch_size)):
        n = np.random.randint(len(train_molecules), size = batch_size)
        x = np.array([train_molecules[i] for i in n])
        l = np.array([train_length[i] for i in n])
        c = np.array([train_labels[i] for i in n])
        cost = model.train(x, l, c)
        train_loss.append(cost)
    
    for iteration in range(int(len(test_molecules)/batch_size)):
        n = np.random.randint(len(test_molecules), size = batch_size)
        x = np.array([test_molecules[i] for i in n])
        l = np.array([test_length[i] for i in n])
        c = np.array([test_labels[i] for i in n])
        cost = model.test(x, l, c)
        test_loss.append(cost)
    
    train_loss = np.mean(np.array(train_loss))        
    test_loss = np.mean(np.array(test_loss))    
    end = time.time()    
    if epoch==0:
        print ('epoch\ttrain_loss\ttest_loss\ttime (s)')
    print ("%s\t%.3f\t%.3f\t%.3f" %(epoch, train_loss, test_loss, end-st))
    ckpt_path = 'save/model.ckpt'
    model.save(ckpt_path, epoch)

