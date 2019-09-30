
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import warnings
warnings.filterwarnings('ignore')

(X_train,y_train),(X_test,y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)

plt.imshow(X_train[0]);
plt.axis('off')
plt.title(y_train[0])
plt.show()

plt.figure(figsize=(10,10))
for i in range(0,20):
  plt.subplot(5,4,i+1)
  plt.imshow(X_train[i])
  plt.title(y_train[i])
  plt.axis('off')
plt.show()

batch_size = 128
epochs = 12
num_classes = 10
img_rows,img_cols = 28,28
if K.image_data_format== 'channels_first':
  X_train = X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
  X_test = X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
  input_shape = (1,img_rows,img_cols)
else:
  X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
  X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
  input_shape = (img_rows,img_cols,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255

print('Number of training samples {}'.format(X_train.shape[0]))
print('number of test samples {}'.format(X_test.shape[0]))
print('training set shape {}'.format(X_train.shape))
print('test set shape {}'.format(X_test.shape))

for i in range(0,5):
  print(y_train[i])

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

print(y_train.shape)

y_train[0]

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,kernel_size = (3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes,activation='softmax'))

model.compile(optimizer = keras.optimizers.adadelta(),loss = keras.losses.categorical_crossentropy,metrics = ['accuracy'])

model.fit(X_train,y_train,batch_size = batch_size,epochs = epochs,verbose=1,validation_data = (X_test,y_test))

score = model.evaluate(X_test,y_test)
print('model loss {}'.format(score[0]))
print('model accuracy {}'.format(score[1]))

model.summary()


