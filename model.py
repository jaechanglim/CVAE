import numpy as np
import tensorflow as tf
import numpy as np
import threading

class CVAE():
    def __init__(self,
                 vocab_size,
                 args
                  ):

        self.vocab_size = vocab_size
        self.batch_size = args.batch_size
        self.latent_size = args.latent_size
        self.lr = tf.Variable(args.lr, trainable=False)
        self.num_prop = args.num_prop
        self.stddev = args.stddev
        self.mean = args.mean
        self.unit_size = args.unit_size
        self.n_rnn_layer = args.n_rnn_layer
        
        self._create_network()


    def _create_network(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size, None])
        self.Y = tf.placeholder(tf.int32, [self.batch_size, None])
        self.C = tf.placeholder(tf.float32, [self.batch_size, self.num_prop])
        self.L = tf.placeholder(tf.int32, [self.batch_size])
        

        
        decoded_rnn_size = [self.unit_size for i in range(self.n_rnn_layer)]
        encoded_rnn_size = [self.unit_size for i in range(self.n_rnn_layer)]
        
        with tf.variable_scope('decode'):
            decode_cell=[]
            for i in decoded_rnn_size[:]:
                decode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
            self.decoder = tf.nn.rnn_cell.MultiRNNCell(decode_cell)
        
        with tf.variable_scope('encode'):
            encode_cell=[]
            for i in encoded_rnn_size[:]:
                encode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell(encode_cell)
        
        self.weights = {}
        self.biases = {}
        self.eps = {
            'eps' : tf.random_normal([self.batch_size, self.latent_size], stddev=self.stddev, mean=self.mean)
        }


        self.weights['softmax'] = tf.get_variable("softmaxw", initializer=tf.random_uniform(shape=[decoded_rnn_size[-1], self.vocab_size], minval = -0.1, maxval = 0.1))       
        
        self.biases['softmax'] =  tf.get_variable("softmaxb", initializer=tf.zeros(shape=[self.vocab_size]))
        self.weights['out_mean'] = tf.get_variable("outmeanw", initializer=tf.contrib.layers.xavier_initializer(), shape=[self.unit_size, self.latent_size]),
        self.weights['out_log_sigma'] = tf.get_variable("outlogsigmaw", initializer=tf.contrib.layers.xavier_initializer(), shape=[self.unit_size, self.latent_size]),
        self.biases['out_mean'] = tf.get_variable("outmeanb", initializer=tf.zeros_initializer(), shape=[self.latent_size]),
        self.biases['out_log_sigma'] = tf.get_variable("outlogsigmab", initializer=tf.zeros_initializer(), shape=[self.latent_size]),

        self.embedding_encode = tf.get_variable(name = 'encode_embedding', shape = [self.latent_size, self.vocab_size], initializer = tf.random_uniform_initializer( minval = -0.1, maxval = 0.1))
        self.embedding_decode = tf.get_variable(name = 'decode_embedding', shape = [self.latent_size, self.vocab_size], initializer = tf.random_uniform_initializer( minval = -0.1, maxval = 0.1))
        
        self.latent_vector, self.mean, self.log_sigma = self.encode()

        self.decoded, decoded_logits = self.decode(self.latent_vector)
        #self.Y_generated = self.generate()

        weights = tf.sequence_mask(self.L, tf.shape(self.X)[1])
        weights = tf.cast(weights, tf.int32)
        weights = tf.cast(weights, tf.float32)
        self.reconstr_loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits=decoded_logits, targets=self.Y, weights=weights))
        self.latent_loss = self.cal_latent_loss(self.mean, self.log_sigma)

        # Loss

        self.loss = self.latent_loss + self.reconstr_loss 
        #self.loss = self.reconstr_loss 
        optimizer    = tf.train.AdamOptimizer(self.lr)
        self.opt = optimizer.minimize(self.loss)
        
        self.mol_pred = tf.argmax(self.decoded, axis=2)
        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=None)
        #tf.train.start_queue_runners(sess=self.sess)
        print ("Network Ready")

    def encode(self): 
        X = tf.nn.embedding_lookup(self.embedding_encode, self.X)
        C = tf.expand_dims(self.C, 1)
        C = tf.tile(C, [1, tf.shape(X)[1], 1])
        inp = tf.concat([X, C], axis=-1)
        _, state = tf.nn.dynamic_rnn(self.encoder, inp, dtype=tf.float32, scope = 'encode', sequence_length = self.L)
        c,h = state[-1]
        self.weights['out_mean'] = tf.reshape(self.weights['out_mean'], [self.unit_size, -1])
        self.weights['out_log_sigma'] = tf.reshape(self.weights['out_log_sigma'], [self.unit_size, -1])
        mean = tf.matmul(h, self.weights['out_mean'])+self.biases['out_mean']
        log_sigma = tf.matmul(h, self.weights['out_log_sigma'])+self.biases['out_log_sigma']
        retval = mean+tf.exp(log_sigma/2.0)*self.eps['eps']
        return retval, mean, log_sigma

    def decode(self, Z):
        seq_length=tf.shape(self.X)[1]
        new_Z = tf.tile(tf.expand_dims(Z, 1), [1, seq_length, 1])
        C = tf.expand_dims(self.C, 1)
        C = tf.tile(C, [1, tf.shape(self.X)[1], 1])
        X = tf.nn.embedding_lookup(self.embedding_encode, self.X)
        inputs = tf.concat([new_Z, X, C], axis=-1)
        self.initial_decoded_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.zeros((self.batch_size, self.unit_size)), tf.zeros((self.batch_size, self.unit_size))) for i in range(3)])
        #self.initial_decoded_state=self.decoder.zero_state() 
        Y, self.output_decoded_state = tf.nn.dynamic_rnn(self.decoder, inputs, dtype=tf.float32, scope = 'decode', sequence_length = self.L, initial_state=self.initial_decoded_state)
        Y = tf.reshape(Y, [self.batch_size*seq_length, -1])
        Y = tf.matmul(Y, self.weights['softmax'])+self.biases['softmax']
        Y_logits = tf.reshape(Y, [self.batch_size, seq_length, -1])
        Y = tf.nn.softmax(Y_logits)
        return Y, Y_logits

    def save(self, ckpt_path, global_step):
        self.saver.save(self.sess, ckpt_path, global_step = global_step)
        #print("model saved to '%s'" % (ckpt_path))

    def assign_lr(self, learning_rate):
        self.sess.run(tf.assign(self.lr, learning_rate ))
    
    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def get_latent_vector(self, x, c, l):
        return self.sess.run(self.latent_vector, feed_dict={self.X : x, self.C : c, self.L : l})

    def cal_latent_loss(self, mean, log_sigma):
        latent_loss = tf.reduce_mean(-0.5*(1+log_sigma-tf.square(mean)-tf.exp(log_sigma)))
        return latent_loss
    
    def train(self, x, y, l, c):
        _, r_loss, l_loss = self.sess.run([self.opt, self.reconstr_loss, self.latent_loss], feed_dict = {self.X :x, self.Y:y, self.L : l, self.C : c})
        return r_loss + l_loss
    
    def test(self, x, y, l, c):
        mol_pred, r_loss, l_loss  = self.sess.run([self.mol_pred, self.reconstr_loss, self.latent_loss], feed_dict = {self.X :x, self.Y:y, self.L : l, self.C : c})
        return r_loss + l_loss

    def sample(self, latent_vector, c, start_codon, seq_length):
        l = np.ones((self.batch_size)).astype(np.int32)
        x=start_codon
        preds = []
        for i in range(seq_length):
            if i==0:
                x, state = self.sess.run([self.mol_pred, self.output_decoded_state], feed_dict = {self.X:x, self.latent_vector:latent_vector, self.L : l, self.C : c})
            else:
                x, state = self.sess.run([self.mol_pred, self.output_decoded_state], feed_dict = {self.X:x, self.latent_vector:latent_vector, self.L : l, self.C : c, self.initial_decoded_state:state})
            preds.append(x)
        return np.concatenate(preds,1).astype(int).squeeze()
