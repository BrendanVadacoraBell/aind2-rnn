import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    #get the length of the series
    series_length = len(series)
    
    #between the window size and the length of the series
    #assign the output to y (the current index)
    #and assign everything before the current index of the window size
    #to X (input)
    for i in range(window_size, series_length):
        y.append(series[i])
        X.append(series[i-window_size:i])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))


    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    import string
    char_set = {char for char in text}

    valid = string.ascii_lowercase
    punctuation = [' ', '!', ',', '.', ':', ';', '?']

    # remove as many non-english characters and character sequences as you can
    def invalidate_char(char):
        return not (char in valid or char in punctuation)

    invalid_char_set = set(filter(invalidate_char, char_set))

    for char in invalid_char_set:
        text = text.replace(char, " ")

    # shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    #get the length of the series
    series_length = len(text)
    
    #between the window size and the length of the series
    #assign the output to outputs (the current index)
    #and assign everything before the current index of the window size
    #to inputs
    for i in range(window_size, series_length, step_size):
        outputs.append(text[i])
        inputs.append(text[i-window_size:i])
    
    return inputs,outputs
