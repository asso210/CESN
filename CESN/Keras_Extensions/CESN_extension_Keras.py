from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Lambda, Concatenate, Dense, Flatten, Conv2D, Conv3D, Conv1D, AveragePooling2D, AveragePooling3D, AveragePooling1D, BatchNormalization, Dropout,MaxPooling3D, MaxPooling2D, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import RandomNormal
from tensorflow import keras
from sklearn.metrics import accuracy_score
import seaborn as sn
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ESN(Layer): 
    def __init__(self, input_dim, output_dim, N_reservoir = 30, seed = None,
                 spectral_radius = 1, noise = 0.001,
                 random_state = None, **kwargs): 
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.N_reservoir = N_reservoir
        self.random_state = random_state
        self.spectral_radius = spectral_radius
        self.noise = noise
        self.seed = seed
        
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand
            
        w_init = tf.random_uniform_initializer(minval=0, maxval=1, seed=self.seed)
        #c_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        self.w_input = tf.Variable(
            initial_value=w_init(shape=(self.input_dim, self.N_reservoir), dtype="float32") * 2 - 1,
            trainable=True,
        )
        self.w_reservoir = tf.Variable(
            initial_value=w_init(shape=(self.N_reservoir, self.N_reservoir), dtype="float32") - 0.5,
            trainable=False,
        )
        self.w_output = tf.Variable(
            initial_value=w_init(shape=(self.N_reservoir, self.N_reservoir), dtype="float32")*2 -1,
            trainable=False,
        )
        self.w_scale_input = tf.Variable(
            initial_value=w_init(shape=(), dtype="float32"),
            trainable=True,
        )


        linaglg_tensor = tf.linalg.eigvals(self.w_reservoir, name=None)
        linaglg_tensor_abs= tf.math.abs(linaglg_tensor, name=None)

        max_index = tf.math.argmax(linaglg_tensor_abs)
        radius = linaglg_tensor_abs[max_index].numpy()
        self.w_reservoir = self.w_reservoir * (self.spectral_radius / radius)
 
        
        super(ESN, self).__init__(**kwargs) 
    def build(self, input_shape): 
        #w_init = tf.random_normal_initializer()

        self.built = True
        #super(MyCustomLayer, self).build(input_shape) # Be sure to call this at the end 
   
    def call(self, input_data):
        step_tensor = []
        a = tf.constant([self.N_reservoir])
        s = tf.shape(input_data)
        s = tf.concat([s, a], axis=0)
        #b = tf.zeros(s, tf.float32)
        b = tf.random.normal(s,mean=0.0,stddev=1.0,dtype=tf.dtypes.float32,seed=None)
        step_tensor.append(b[:,0,0,:])
        
        a = tf.constant([self.N_reservoir])
        s = tf.shape(input_data)
        s = tf.concat([s, a], axis=0)
        #b = tf.zeros(s, tf.float32)
        b = tf.random.normal(s,mean=0.0,stddev=1.0,dtype=tf.dtypes.float32,seed=None)
        states = b[:,0,0,:]
        
        g = tf.constant([self.N_reservoir])
        h = tf.shape(input_data)
        h = tf.concat([h, g], axis=0)
        #b = tf.zeros(s, tf.float32)
        h = tf.random.normal(s,mean=0.0,stddev=1.0,dtype=tf.dtypes.float32,seed=None)
        random_noise = h[:,0,0,:]
        for i in range(1,input_data.shape[1]):
            
            preactivation = tf.matmul(states, self.w_reservoir) +  tf.math.scalar_mul(self.w_scale_input, tf.matmul(input_data[:,i,:], self.w_input))

            post_activation = tf.tanh(preactivation) + self.noise * random_noise
            states = post_activation
            step_tensor.append(post_activation)
        
        final_tensor = tf.stack(step_tensor, axis=1)
        final_tensor = tf.expand_dims(final_tensor, axis=3)
        #output_tensor = tf.tanh(tf.matmul(final_tensor,self.w_output))#tf.keras.activations.relu(tf.matmul(final_tensor,self.w_output))
        #output_tensor = tf.expand_dims(output_tensor, axis=3)
        return final_tensor

    def compute_output_shape(self, input_shape): 

        return (None, input_shape[1], self.output_dim)
