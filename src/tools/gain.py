'''
GAIN function
'''
import numpy as np
import tensorflow as tf
import math
from tensorflow import math
from tensorflow import keras
from tqdm import tqdm

from .utils import normalization, renormalization, rounding
from .utils import binary_sampler, uniform_sampler, sample_batch_index

def gain(data_x, gain_parameters):
    '''
    Impute missing values in data_x

    Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations

    Returns:
    - imputed_data: imputed data
    '''

    # Define mask matrix
    data_m = 1-np.isnan(data_x)

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # Other parameters
    no, dim = data_x.shape

    # Hidden state dimensions
    h_dim = int(dim)

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    ## Layers
    # Dense ReLu
    def dense(inputs, weights):
        # try leaky_relu too
        return tf.nn.relu(tf.matmul(inputs, weights))
    
    # Dense sigmoid
    def dense_sigmoid(inputs, weights):
        return tf.nn.sigmoid(tf.matmul(inputs, weights))
    
    ## Weights
    #xavier init
    initializer = tf.initializers.glorot_uniform()

    def get_weight(shape, name):
        #trainable = True enables the tf.variable to be differentiable 
        return tf.Variable(initializer(shape), name=name, trainable=True, dtype=tf.float32)

    shapes = [[dim*2, dim], [dim, dim], [dim, dim]]
    weights_G = []
    weights_D = []
    for i in range(len(shapes)):
        weights_G.append(get_weight(shapes[i] , f"weight{i}"))
        weights_D.append(get_weight(shapes[i] , f"weight{i}"))
    

    ## GAIN architecture
    # generator
    def generator(x, m):
        # concat data and mask
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = dense(inputs=inputs, weights=weights_G[0])
        G_h2 = dense(inputs=G_h1, weights=weights_G[1])
        # MinMax normalized output
        G_prob = dense_sigmoid(inputs=G_h2, weights=weights_G[2])
        
        return G_prob

    # discriminator
    def discriminator(x, h):
        # concat data and hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = dense(inputs=inputs, weights=weights_D[0])
        D_h2 = dense(inputs=D_h1, weights=weights_D[1])
        D_prob = dense_sigmoid(inputs=D_h2, weights=weights_D[2])

        return D_prob

    ## GAIN loss
    def D_loss(D_prob, m): 
        loss = -tf.reduce_mean(m * tf.math.log(D_prob + 1e-8) + (1-m) * tf.math.log(1. - D_prob + 1e-8))
        
        return loss


    def G_loss(D_prob, G_prob, x, m):
        # L_M loss
        MSE_loss = tf.reduce_mean((m * x - m * G_prob)**2 / tf.reduce_mean(m))
        #L_G loss
        loss_temp = -tf.reduce_mean((1-m) * tf.math.log(D_prob + 1e-8))
        
        # punish MSE_loss more with alpha
        loss = loss_temp + alpha * MSE_loss
        
        return loss

    ## training
    # sampling from the data
    def sample_data(norm_data_x, data_m):
        # sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        # sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        #sample hint vector
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp
        
        X_mb = X_mb.astype('float32')
        M_mb = M_mb.astype('float32')
        H_mb = H_mb.astype('float32')
        Z_mb = Z_mb.astype('float32')

        # combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

        return X_mb, M_mb, H_mb
    
    # training discriminator
    optimizer = tf.optimizers.Adam()

    def D_train_step(x, m, h):
        with tf.GradientTape() as tape:
            D_prob = discriminator(x, h)
            current_loss = D_loss(D_prob, m)
            grads = tape.gradient(current_loss, weights_D)
            optimizer.apply_gradients(zip(grads, weights_D))

    # training generator
    def G_train_step(x, m, h):
        with tf.GradientTape() as tape:
            G_prob = generator(x, m)
            D_prob = discriminator(x, h)
            current_loss = G_loss(D_prob, G_prob, x, m)
            grads = tape.gradient(current_loss, weights_G)
            optimizer.apply_gradients(zip(grads, weights_G))


    # iterations
    for it in tqdm(range(iterations)):

        #sample batch
        X_mb, M_mb, H_mb = sample_data(norm_data_x, data_m)
        
        D_train_step(X_mb, M_mb, H_mb)

        G_train_step(X_mb, M_mb, H_mb)

    
    # return imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x

    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

    X_mb = X_mb.astype('float32')
    M_mb = M_mb.astype('float32')

    imputed_data = generator(X_mb, M_mb)

    imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
    imputed_data = imputed_data.numpy()

    # renormalize
    imputed_data = renormalization(imputed_data, norm_parameters)

    # rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data
        
                        
