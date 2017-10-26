import matplotlib.pyplot as plt
import numpy as np
import time
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()

classes = [8, 5] # Ship, Dog
ind0_train = np.where(train_labels.flatten() == classes[0])[0]
ind1_train = np.where(train_labels.flatten() == classes[1])[0]
ind0_test = np.where(test_labels.flatten() == classes[0])[0]
ind1_test = np.where(test_labels.flatten() == classes[1])[0]

train_data0 = train_features[ind0_train]
train_data1 = train_features[ind1_train]
test_data0 = test_features[ind0_test]
test_data1 = test_features[ind1_test]

train_data = np.concatenate((train_data0, train_data1))
train_labels = np.concatenate((np.zeros(train_data0.shape[0]), np.zeros(train_data1.shape[0]) + 1))
test_data = np.concatenate((test_data0, test_data1))
test_labels = np.concatenate((np.zeros(test_data0.shape[0]), np.zeros(test_data1.shape[0]) + 1))

perm_train = np.random.permutation(train_data.shape[0])
perm_test = np.random.permutation(test_data.shape[0])
train_data = train_data[perm_train]
train_labels = train_labels[perm_train]
test_data = test_data[perm_test]
test_labels = test_labels[perm_test]

def show_rand_class(cls):
	ind = np.random.choice(np.where(train_labels == cls)[0])
	plt.imshow(train_data[ind])
	plt.show

cnn_train = train_data.astype('float32')/255
cnn_train_labels = np_utils.to_categorical(train_labels, 2)
cnn_test = test_data.astype('float32')/255
cnn_test_labels = np_utils.to_categorical(test_labels, 2)

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=cnn_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


start = time.time()
model_info = model.fit(cnn_train, cnn_train_labels, 
                       batch_size=256, epochs=20, 
                       validation_data = (cnn_train, cnn_train_labels))
end = time.time()

print("Model took %0.2f seconds to train"%(end - start))
print("Accuracy on test data is: " + str(accuracy(cnn_test, cnn_test_labels, model)))


dr_train = np.sum(train_data.astype('float32')/255, axis = 3).reshape(train_data.shape[0],1024)
dr_train_labels = np_utils.to_categorical(train_labels, 2)
dr_test = np.sum(test_data.astype('float32')/255, axis = 3).reshape(test_data.shape[0], 1024)
dr_test_labels = np_utils.to_categorical(test_labels, 2)

start = time.time()

cov = np.matmul(dr_train.T, dr_train)
D, V = np.linalg.eig(cov)
eig_inds = np.argsort(-D)
D = D[eig_inds]
V = V[:, eig_inds]
top = np.asarray([i for i in range(25)])
D1 = D[top]
V1 = V[:, top]
print("Preserved Variance: " + str(np.sum(D1)/np.sum(D)))

t_train = np.matmul(dr_train, V1)
t_test = np.matmul(dr_test, V1)

model2 = Sequential()

model2.add(Dense(50, input_dim = 25, activation = 'relu'))
model2.add(Dense(25, activation = 'relu'))
model2.add(Dense(2, activation = 'sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model2.fit(t_train, dr_train_labels, epochs = 200, batch_size = 1024)

end = time.time()

print("Model took %0.2f seconds to train"%(end - start))
print("Accuracy on test data is: "+ str(accuracy(t_test, dr_test_labels, model2)))
