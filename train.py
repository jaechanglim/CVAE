from model import CVAE
from utils import *
import numpy as np
import os
import tensorflow as tf
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--unit_size', help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--n_rnn_layer', help='number of rnn layer', type=int, default=3)
parser.add_argument('--seq_length', help='max_seq_length', type=int, default=120)
parser.add_argument('--prop_file', help='name of property file', type=str)
parser.add_argument('--mean', help='mean of VAE', type=float, default=0.0)
parser.add_argument('--stddev', help='stddev of VAE', type=float, default=1.0)
parser.add_argument('--num_epochs', help='epochs', type=int, default=100)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
parser.add_argument('--num_prop', help='number of propertoes', type=int, default=3)
parser.add_argument('--save_dir', help='save dir', type=str, default='save/')
args = parser.parse_args()

print (args)
print("Num GPUs Available: ", len(tf.compat.v1.config.list_physical_devices('GPU')))
#convert smiles to numpy array
molecules_input, molecules_output, char, vocab, labels, length = load_data(args.prop_file, args.seq_length)
vocab_size = len(char)

#make save_dir
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

#divide data into training and test set
num_train_data = int(len(molecules_input)*0.75)
train_molecules_input = molecules_input[0:num_train_data]
test_molecules_input = molecules_input[num_train_data:-1]

train_molecules_output = molecules_output[0:num_train_data]
test_molecules_output = molecules_output[num_train_data:-1]

train_labels = labels[0:num_train_data]
test_labels = labels[num_train_data:-1]

train_length = length[0:num_train_data]
test_length = length[num_train_data:-1]

model = CVAE(vocab_size,
             args
             )
print ('Number of parameters : ', np.sum([np.prod(v.shape) for v in tf.compat.v1.trainable_variables()]))

for epoch in range(args.num_epochs):

    st = time.time()
    # Learning rate scheduling 
    #model.assign_lr(learning_rate * (decay_rate ** epoch))
    train_loss = []
    test_loss = []
    st = time.time()
    for iteration in range(len(train_molecules_input)//args.batch_size):
        n = np.random.randint(len(train_molecules_input), size = args.batch_size)
        x = np.array([train_molecules_input[i] for i in n])
        y = np.array([train_molecules_output[i] for i in n])
        l = np.array([train_length[i] for i in n])
        c = np.array([train_labels[i] for i in n])
        cost = model.train(x, y, l, c)
        train_loss.append(cost)
    
    for iteration in range(len(test_molecules_input)//args.batch_size):
        n = np.random.randint(len(test_molecules_input), size = args.batch_size)
        x = np.array([test_molecules_input[i] for i in n])
        y = np.array([test_molecules_output[i] for i in n])
        l = np.array([test_length[i] for i in n])
        c = np.array([test_labels[i] for i in n])
        cost = model.test(x, y, l, c)
        test_loss.append(cost)
    
    train_loss = np.mean(np.array(train_loss))        
    test_loss = np.mean(np.array(test_loss))    
    end = time.time()    
    if epoch==0:
        print ('epoch\ttrain_loss\ttest_loss\ttime (s)')
    print ("%s\t%.3f\t%.3f\t%.3f" %(epoch, train_loss, test_loss, end-st))
    ckpt_path = args.save_dir+'/model_'+str(epoch)+'.ckpt'
    model.save(ckpt_path, epoch)

