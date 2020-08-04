import tensorflow.keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import tensorflow
# from tensorflow import set_random_seed DONT NEED THIS FOR TENSORFLOW2.0
import os
#  Just disables the warning, doesn't enable AVX/FMA
# I can ignore this cuz i have a gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# sets the random seed for the numpy random generator and tensorflow backend.
def seedy(s):
    np.random.seed(s)
    #set_random_seed(s)
    tensorflow.random.set_seed(s)

# define template for model
class AutoEncoder:
    def __init__(self, encoding_dim=3):
        self.encoding_dim = encoding_dim # overall dimension that we are reducing our numbers down to.
        r = lambda: np.random.randint(1,3)
        self.x = np.array([[r(),r(), r()] for _ in range(1000)]) # create 1000 arrays of 3 random ints as training data
        print(self.x)

    # encoder creates a neuron for each input
    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape)) # we set the shape of our training example to the first training input we give it.
        encoded = Dense(self.encoding_dim, activation='relu')(inputs) # encoding_dim number of neurons
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(3)(inputs)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    # concatenate encoder and decoder
    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()

        # combine both encoder and decoder models
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model
    
    # compiles top model using stochastic gradient descent.
    def fit(self, batch_size=10, epochs=300):
        # mse == mean squared error
        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = '.\\log\\'
        print(log_dir)
        # saves the loss to logs so we can visualize it later.
        tbCallBack = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                            histogram_freq=0,
                                                            write_graph=True,
                                                            write_images=True)
        # input is self.x, and our target is self.x as well.
        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[tbCallBack])

    # saves the weight of our models. 
    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
            print("HELLO")
        else:
            self.encoder.save(r'./weights/encoder_weights.h5')
            self.decoder.save(r'./weights/decoder_weights.h5')
            self.model.save(r'./weights/ae_weights.h5')

if __name__ == '__main__':
    seedy(2)
    ae = AutoEncoder(encoding_dim=2)
    ae.encoder_decoder()
    # fit data set (input) to the Neural Network. 50 items at a time, 300 times total
    ae.fit(batch_size=50, epochs=300) # one epoch is the whole process of going through the encoding and getting the loss.
    ae.save()