import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras import regularizers
import optuna

class CoSTCo(tfk.Model):
    def __init__(self, rank = 10, 
                       shape = (1,1,1),
                       filters = 64,
                       reg = 0):
        '''
        rank is an important hyperparamater for tensor factorization 
        a higher rank means the model is more powerful but more likely to overfit
        shape is just the shape of the input tensor
        filters is the number of conv filters
        reg is a weight on the l2 regularization of the factor matrices
        '''

        super(CoSTCo, self).__init__()


  
        # C_n are the paramater matrices for each axis of the tensor
        # For example C_2 is a matrix of size NUM_WEBSITES by Rank
        # Ex. if you take the 10th row of C_2, it describes the 10th website
        # you can think of this as similar to the latent space of an autoencoder.
        self.C_1 = tfkl.Embedding(input_dim = shape[0], 
                                  output_dim=rank,
                                  embeddings_initializer='normal',
                                  trainable = True,
                                  activity_regularizer = regularizers.l2(reg) )
        self.C_2 = tfkl.Embedding(input_dim = shape[1],
                                  output_dim=rank,
                                  embeddings_initializer='normal',
                                  trainable = True,
                                  activity_regularizer = regularizers.l2(reg) )
        self.C_3 = tfkl.Embedding(input_dim = shape[2],
                                  output_dim=rank,
                                  embeddings_initializer='normal',
                                  trainable = True,
                                  activity_regularizer = regularizers.l2(reg) )


        # These are the layers that make CoSTCo a non-linear (deep) tensor completion method
        self.conv1 = tfkl.Conv2D(filters,kernel_size = (1,3),activation = 'relu',trainable = True)
        self.b1 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(filters,kernel_size = (rank,1),activation = 'relu',trainable = True)
        self.b2 = tfkl.BatchNormalization()
        self.dense1 = tfkl.Dense(filters,
                                 activation = 'relu',
                                 trainable = True)    
        self.b3 = tfkl.BatchNormalization()
        self.dense2 = tfkl.Dense(1,trainable = True)

    def call(self,indices):
        '''
        indices is a three-tuple of indices into the tensor
        such as (1,2,3)

        We will return a number between 0-5 representing the prediction for that index
        of the tensor
        Ex. if indices is (1,2,3) this function will return the predicted review rating that the 1st device
        left on the 2nd website for the 3rd item
        '''

        # select the appropiate rows of the factor matrices
        # c_1,c_2 and c_3 will be sets of vectors with the length of rank
        c_1 = self.C_1(indices[...,0])
        c_2 = self.C_2(indices[...,1])
        c_3 = self.C_3(indices[...,2])

        # Because all the vectors are of the same size, we can stack them into a matrix
        # of size Rank X 3
        c = tf.stack([c_1,c_2,c_3],axis = -1)
  
        # Turn this into size Rank X 3 X 1, we will treat this last dimension as the filter dimension
        c = tf.expand_dims(c,axis = -1)

        # Apply the first convolution to this matrix, reducing it to size Rank X 1 X filters
        c = self.conv1(c)
        c = self.b1(c)        

        # Apply the first convolution to this matrix, reducing it to size 1 X 1 X filters
        c = self.conv2(c)
        c = self.b2(c)        
  
        # Flatten it to size filters
        c = tfkl.Flatten()(c)

        # Apply one dense layer, resulting in a size of filters
        c = self.dense1(c)
        c = self.b3(c)        

        # Apply a dense layer reducing it to a scalar between 0 and 5
        c = self.dense2(c)

        return c


class CP_WOPT(tfk.Model):
    def __init__(self, rank = 10, 
                       shape = (1,1,1)):
        '''
        rank is an important hyperparamater for tensor factorization 
        a higher rank means the model is more powerful but more likely to overfit
        shape is just the shape of the input tensor
        '''

        super(CP_WOPT, self).__init__()

        # C_n are the paramater matrices for each axis of the tensor
        # For example C_2 is a matrix of size NUM_WEBSITES by Rank
        # Ex. if you take the 10th row of C_2, it describes the 10th website
        # you can think of this as similar to the latent space of an autoencoder.
        self.C_1 = tfkl.Embedding(input_dim = shape[0], output_dim=rank, embeddings_initializer='normal',trainable = True)
        self.C_2 = tfkl.Embedding(input_dim = shape[1], output_dim=rank, embeddings_initializer='normal',trainable = True)
        self.C_3 = tfkl.Embedding(input_dim = shape[2], output_dim=rank, embeddings_initializer='normal',trainable = True)


    def call(self,indices):
        '''
        indices is a three-tuple of indices into the tensor
        such as (1,2,3)

        We will return a number between 0-5 representing the prediction for that index
        of the tensor
        Ex. if indices is (1,2,3) this function will return the predicted review rating that the 1st device
        left on the 2nd website for the 3rd item
        '''

        # select the appropiate rows of the factor matrices
        # c_1,c_2 and c_3 will be sets of vectors with the length of rank
        c_1 = self.C_1(indices[...,0])
        c_2 = self.C_2(indices[...,1])
        c_3 = self.C_3(indices[...,2])


        out = tf.math.reduce_sum(c_1*c_2*c_3,axis = -1)
        return out
