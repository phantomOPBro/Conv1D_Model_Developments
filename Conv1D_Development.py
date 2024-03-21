from keras.layers import Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, concatenate
from keras.models import Model
from keras.layers import MaxPooling1D
import pandas as pd
import random
from datetime import datetime, timedelta


df = pd.read_csv('file.csv')

'''
def generate_features(df):
    # List of date formats
    date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%y-%m-%d', '%d-%m-%y', '%m-%d-%y']
    # Generate 1000 random dates
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2030, 12, 31)
    dates = [(start_date + (end_date - start_date) * random.random()).date() for _ in range(1000)]
    # Format dates and append to dataframe
    for date in dates:
        format = random.choice(date_formats)  # Choose a random date format
        feature = date.strftime(format) + '.pdf'  # Append '.pdf' to the file name
        df.loc[len(df)] = [feature, 0]  # Append new row to dataframe
    return df
'''

# Use the function

df2 = generate_features(df2)

df = pd.concat([df, df2], ignore_index=True)

# Define the alphabet and the input and output sizes
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
input_size = 100
output_size = 14

# Create a tokenizer that converts characters to indices
tokenizer = Tokenizer(num_words=alphabet_size + 1, char_level=True)
tokenizer.fit_on_texts(alphabet)

X = tokenizer.texts_to_sequences(df['feature'])
# Pad or truncate the rows to have the same length
X = pad_sequences(X, maxlen=input_size)
# Convert the label column to a binary vector
y = df['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_val = X_val.astype('float32')
y_val = y_val.astype('float32')


# Define the model parameters
vocab_size = 100  # The size of the vocabulary
max_len = 100  # The maximum length of the input sequence
embed_dim = 10  # The dimension of the embedding
num_filters = 2  # The number of filters for the convolutional layer
kernel_size = 12  # Reduced kernel size
pool_size = 2  # Pool size for MaxPooling1D
hidden_dim1 = 32  # The dimension of the hidden layer
hidden_dim2 = 10  # The dimension of the hidden layer



model = Sequential()
model.add(Embedding(alphabet_size + 1, embed_dim, input_length=max_len))
model.add(Conv1D(num_filters, kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Dropout(0.2))  # add dropout
model.add(Conv1D(num_filters, kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(num_filters, kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dim1, activation='relu'))
model.add(Dropout(0.2))  # add dropout
model.add(Dense(hidden_dim2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

checkpoint = ModelCheckpoint('Con1D_Net_1.h5', verbose=1, monitor='val_loss',
                             save_best_only=True,
                             mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model on the training set and validate on the validation set
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                    callbacks=[checkpoint, early_stopping])

# Evaluate the model on the test set
arr = model.evaluate(X_test, y_test)

loss = arr[0]
accuracy = arr[1]

# Open the file in write mode and save the results
with open('Con1D_Net.txt', 'w') as file:
    file.write(f'Loss: {loss}\n')
    file.write(f'Accuracy: {accuracy}\n')

import pickle

# Save the tokenizer
with open('Con1D_Net.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('Con1D_Net_model_history.csv')

# Save the model
model.save('Con1D_Net_model.h5')







'''

#load

from keras.models import load_model
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model('Con1D_Net_model.h5')

# Load the tokenizer
with open('Con1D_Net_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

input_size = 100

# The feature to predict
input = 'input_to_test'

# Preprocess the feature
input = input.replace(' ', '').lower()
input = np.array(tokenizer.texts_to_sequences([input]))

# Pad the feature
input_padded = pad_sequences(input, maxlen=input_size)

# Make the prediction
prediction = model.predict(input_padded)
print(prediction)


'''

'''
pd.set_option('display.float_format', lambda x: '%.10f' % x)


import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model

# Load the tokenizer
with open('Con1D_Net_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = load_model('Con1D_Net_model.h5')

# Preprocess the new dataframe column
df3['feature'] = df3[''].str.replace(' ', '')
df3['feature'] = df3['feature'].str.lower()
df3['feature'] = df3['feature'].astype(str)

# Tokenize and pad the sequences
X_new = tokenizer.texts_to_sequences(df3['feature'])
X_new = pad_sequences(X_new, maxlen=1014)

# Make predictions
predictions = model.predict(X_new)

# Add the predictions back to the dataframe
df3['predictions'] = predictions



import onnxruntime as ort

# Start an inference session with ONNX Runtime
sess = ort.InferenceSession('Cov_Net_model.onnx')

# The input name for the ONNX model
input_name = sess.get_inputs()[0].name

# Make predictions
predictions = sess.run(None, {input_name: X_new.astype(np.float32)})[0]

# Add the predictions back to the dataframe
df3['predictions'] = predictions



#actually save the model as Onxx

model.save('model.h5')
import tensorflow as tf
loaded_model = tf.keras.models.load_model('model.h5')
tf.saved_model.save(loaded_model, 'saved_model')
!python -m tf2onnx.convert --saved-model 'saved_model' --output 'model.onnx'






#actually load and predict using model

import onnxruntime

# Load the ONNX model
sess = onnxruntime.InferenceSession("Conv1D_model.onnx")

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle

with open('CovNet_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

feature = "15-12228-01 AVC2T45.pdf"

# Tokenize the feature
tokenized = tokenizer.texts_to_sequences([feature])

# Pad the tokenized feature
padded = pad_sequences(tokenized, maxlen=1014)

# Convert the padded feature to a numpy array
input_data = np.array(padded).astype(np.float32)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name

# Get the output name for the ONNX model
output_name = sess.get_outputs()[0].name

# Use the ONNX model to make a prediction
result = sess.run([output_name], {input_name: input_data})

# Print the prediction
print("Prediction:", result[0][0][0])


'''

'''
Fine Tune the model

from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model
model = load_model('Cov_Net_model.h5')

# Freeze all the layers except the last two
for layer in model.layers[:-2]: # Change this to select more/less layers
    layer.trainable = False

# Check the trainable status of each layer
for layer in model.layers:
    print(layer, layer.trainable)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the tokenizer
with open('CovNet_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

new_df = pd.read_csv('file.csv')
# New data preparation
# Assuming `new_df` is your new dataframe and the text is in the 'feature' column
new_df['feature'] = new_df['feature'].str.replace(' ', '')
new_df['feature'] = new_df['feature'].str.lower()

# Tokenize and pad the sequences
X_new = tokenizer.texts_to_sequences(new_df['feature'])
X_new = pad_sequences(X_new, maxlen=1014)

# Assuming 'label' is the label for new data
y_new = new_df['label'].values

# Train the model on the new data
history = model.fit(X_new, y_new, batch_size=128, epochs=10) # Adjust the epochs and batch_size accordingly



loss = arr[0]
accuracy = arr[1]

# Open the file in write mode and save the results
with open('Cov_Net_2ndBranch_lossacc.txt', 'w') as file:
    file.write(f'Loss: {loss}\n')
    file.write(f'Accuracy: {accuracy}\n')





import pandas as pd
import random
from datetime import datetime, timedelta

def generate_features(df):
    # List of date formats
    date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%y-%m-%d', '%d-%m-%y', '%m-%d-%y']
    # Generate 1000 random dates
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2030, 12, 31)
    dates = [(start_date + (end_date - start_date) * random.random()).date() for _ in range(1000)] 
    # Format dates and append to dataframe
    for date in dates:
        format = random.choice(date_formats)  # Choose a random date format
        feature = date.strftime(format) + '.pdf'  # Append '.pdf' to the file name
        df.loc[len(df)] = [feature, 0]  # Append new row to dataframe
    return df


# Use the function
df2 = pd.DataFrame(columns=['feature', 'label'])
df2 = generate_features(df2)

df = pd.concat([df,df2], ignore_index=True)





import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model


df = pd.read_csv('file.csv')
df['feature'] = df['feature'].str.replace(' ', '').str.lower()
input_size = 100
X_test_file = pad_sequences(tokenizer.texts_to_sequences(df['feature']), maxlen=input_size)
y_test = df['label']

X_test_file = X_test_file.astype('float32')
y_test = y_test.astype('float32')

arr = model.evaluate(X_test_file, y_test)



def predict_this(feature):
    feature = feature.replace(' ', '').lower()
    feature = np.array(tokenizer.texts_to_sequences([feature]))
    feature_padded = pad_sequences(feature, maxlen=input_size)
    prediction = model.predict(feature_padded)
    return prediction


'''


'''
Can also do some combination of features


from keras.layers import Flatten
import pandas as pd
from keras.layers import Embedding, Conv1D, Dense
from keras.layers import Flatten
from keras.layers import Input, concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.layers import Flatten
import pandas as pd
import pandas as pd
from keras.layers import Embedding, Conv1D, Dense
from keras.layers import Flatten
from keras.layers import Input, concatenate
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import random
from datetime import datetime, timedelta


df = pd.read_csv('file.csv')

def generate_features(df):
    # List of date formats
    date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%y-%m-%d', '%d-%m-%y', '%m-%d-%y']
    # Generate 1000 random dates
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2030, 12, 31)
    dates = [(start_date + (end_date - start_date) * random.random()).date() for _ in range(1000)]
    # Format dates and append to dataframe
    for date in dates:
        format = random.choice(date_formats)  # Choose a random date format
        feature = date.strftime(format) + '.pdf'  # Append '.pdf' to the file name
        df.loc[len(df)] = [feature, 0]  # Append new row to dataframe
    return df


# Use the function
df2 = pd.DataFrame(columns=['feature', 'label'])
df2 = generate_features(df2)

df = pd.concat([df, df2], ignore_index=True)

df['extension'] = df['feature'].apply(lambda x: x.split('.')[-1] if '.' in x else '')
df['feature'] = df['feature'].apply(lambda x: '.'.join(x.split('.')[:-1]) if '.' in x else x)
df['feature'] = df['feature'].str.replace(' ', '').str.lower()

df['feature_length'] = df['feature'].apply(lambda x: len(x.split('.')[0]))


# Alphabet and sizes
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
input_size = 100
output_size = 14

# Tokenizers
tokenizer_file = Tokenizer(num_words=len(alphabet) + 1, char_level=True)
tokenizer_file.fit_on_texts(df['feature'])
X_file = pad_sequences(tokenizer_file.texts_to_sequences(df['feature']), maxlen=input_size)

tokenizer_ext = Tokenizer(num_words=len(alphabet) + 1, char_level=True)
tokenizer_ext.fit_on_texts(df['extension'])
X_ext = pad_sequences(tokenizer_ext.texts_to_sequences(df['extension']), maxlen=input_size)

# Normalize feature lengths
df['feature_length'] /= df['feature_length'].max()
X_lengths = df['feature_length'].values.reshape(-1, 1)

# Target variable
y = df['label'].values

# Train-test split
X_train_file, X_test_file, y_train, y_test = train_test_split(X_file, y, test_size=0.2, random_state=42)
X_train_ext, X_test_ext, _ , _ = train_test_split(X_ext, y, test_size=0.2, random_state=42)
X_train_lengths, X_test_lengths = train_test_split(X_lengths, test_size=0.2, random_state=42)

# Convert to float
X_train_file = X_train_file.astype('float32')
X_train_ext = X_train_ext.astype('float32')
X_train_lengths = X_train_lengths.astype('float32')
y_train = y_train.astype('float32')

# Model parameters
embed_dim = 100  # Embedding dimension
num_filters = 32  # Number of filters for the convolutional layer
kernel_size = 12  # Kernel size
hidden_dim = 8  # Hidden layer dimension
hidden_dim1  = 16
hidden_dim2 = 8
hidden_dim3 = 4

# feature sequences model
input_file = Input(shape=(input_size,), dtype='int32')
x = Embedding(alphabet_size + 1, embed_dim, input_length=input_size)(input_file)
x = Conv1D(num_filters, kernel_size, activation='relu')(x)
x = Flatten()(x)
x = Dense(hidden_dim, activation='relu')(x)
seq_model = Model(input_file, x)

# feature length model
length_input = Input(shape=(1,))
length_dense = Dense(hidden_dim, activation='relu')(length_input)
length_model = Model(length_input, length_dense)

# File extension model
input_ext = Input(shape=(input_size,), dtype='int32')
x_ext = Embedding(alphabet_size + 1, embed_dim, input_length=input_size)(input_ext)
x_ext = Flatten()(x_ext)
x_ext = Dense(hidden_dim, activation='relu')(x_ext)
ext_model = Model(input_ext, x_ext)

# Concatenate outputs of the three models
concatenated = concatenate([seq_model.output, length_model.output, ext_model.output])

# Add final dense layers
hidden1 = Dense(hidden_dim1, activation='relu')(concat)
hidden2 = Dense(hidden_dim2, activation='relu')(hidden1)
hidden3 = Dense(hidden_dim3, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(x)

# Final model
final_model = Model(inputs=[seq_model.input, length_model.input, ext_model.input], outputs=output)

# Compile model
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('combined_feaures_model-{epoch:03d}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train model
history = final_model.fit([X_train_file, X_train_lengths, X_train_ext], y_train, batch_size=128, epochs=100,
                          validation_data=([X_test_file, X_test_lengths, X_test_ext], y_test),
                          callbacks=[checkpoint, early_stopping])

# Evaluate the final model on the test set
arr = final_model.evaluate([X_test_file, X_test_lengths, X_test_ext], y_test)


loss = arr[0]
accuracy = arr[1]

# Open the file in write mode and save the results
with open('combined_feaures_model_lossacc.txt', 'w') as file:
    file.write(f'Loss: {loss}\n')
    file.write(f'Accuracy: {accuracy}\n')

history_df = pd.DataFrame(history.history)
history_df.to_csv('combined_feaures_model_history.csv')

# Save the final model
final_model.save('combined_feaures_model_model.h5')

# Save the tokenizer
with open('combined_feaures_model_tokenizer_file.pickle', 'wb') as handle:
    pickle.dump(tokenizer_file, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Save the tokenizer
with open('combined_feaures_model_tokenizer_ext.pickle', 'wb') as handle:
    pickle.dump(tokenizer_ext, handle, protocol=pickle.HIGHEST_PROTOCOL)



'''