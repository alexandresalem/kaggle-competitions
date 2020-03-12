import pandas as pd
import numpy as np
from keras import layers,models,optimizers
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.callbacks import TensorBoard

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

turn = 7
train_data = pd.read_csv('train.csv')
train_features = train_data.iloc[:,1:]
train_labels = train_data['label']

tamanho = len(train_data)

import time
train_features = train_features.to_numpy()
train_features = np.reshape(train_features, (tamanho,28,28,1))
train_features = train_features.astype('float32')/255
train_labels = train_labels.to_numpy().astype('float32')
train_labels = to_categorical(train_labels)

name = f'Digit-Recognition-CNN-128x2-256-128-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{name}')

def build_model():

    model = models.Sequential()
    model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(28,28,1), kernel_constraint=maxnorm(3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3, 3), kernel_constraint=maxnorm(3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(256, (3, 3),kernel_constraint=maxnorm(3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())

    model.add(layers.Dense(128,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['acc'])

    return model



k = 5
num_val_samples = tamanho//k
num_epochs = 5
all_scores = []

for i in range(k):
    val_features = train_features[i*num_val_samples:(i+1)*num_val_samples]
    val_labels = train_labels[i*num_val_samples:(i+1) * num_val_samples]

    partial_train_features = np.concatenate([
        train_features[:i*num_val_samples],
        train_features[(i+1)*num_val_samples:]],axis=0)
    partial_train_labels = np.concatenate([
        train_labels[:i*num_val_samples],
        train_labels[(i + 1) * num_val_samples:]],axis=0)
    model = build_model()
    history = model.fit(partial_train_features,partial_train_labels,epochs=num_epochs,batch_size=32, validation_data=(val_features,val_labels), callbacks=[tensorboard])



test_data = pd.read_csv('test.csv')
test_features = test_data.to_numpy()
preview_features = np.reshape(test_features,(len(test_data),28,28))
test_features = np.reshape(test_features, (len(test_data),28,28,1))

# plt.imshow(preview_features[3])
# plt.title('0 ou 9')
# plt.show()

test_features = test_features.astype('float32')/255

model = build_model()
history = model.fit(train_features,train_labels,epochs=10,verbose=1,batch_size=128)



# serialize model to JSON
model_json = model.to_json()
with open(f"model{turn}.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(f"model{turn}.h5")

print("Saved model to disk")

# load json and create model
json_file = open(f'model{turn}.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(f"model{turn}.h5")
print("Loaded model from disk")


results = loaded_model.predict(test_features)
numbers = []
index = []
i = 1
for result in results:
    number = np.argmax(result)
    numbers.append(number)
    index.append(i)
    i += 1


df = pd.DataFrame()
df['ImageId'] = index
df['Label'] = numbers

df.to_csv(r'submission7.csv',index=False,header=True)



