from keras.layers import Dense, Dropout,Embedding, Activation
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import SimpleRNN
from keras.datasets import reuters

#set size of data we want to actually use from the dataset. We're using the top 100 words from the data in the max_features
#subset from the reuters dataset.
max_features = 30000
max_length = 100
batch_size = 48

#keras makes it very easy to load our data!
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)
#we need to pad the sequences to get the right shape
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

#keras makes model building a lot easier too!
model = Sequential()
#we need to add an embedding that will hold all of our data.
model.add(Embedding(max_features, 256))
#this is the step that is crucial to everything, it loads the RNN layer instead of LSTM layer into the model with the correct parameters, note
#we use the same size as the second dimm of our embedding
model.add(SimpleRNN(256, dropout=0.2, recurrent_dropout=0.2))
#add the activation layer
model.add(Dense(1, activation='sigmoid'))

##using MSE and Adam for loss and optimizer respectively
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
#I'm using a high number of epochs because my first few runs went badly
model.fit(x_train, y_train,batch_size=batch_size,epochs=25,validation_data=(x_test, y_test))
#obtain accuracy of LTSM to compare with rnn
tempacc = model.evaluate(x_test, y_test,batch_size=batch_size)
print('Test accuracy:'+ str(tempacc))