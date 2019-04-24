####################################################
# 
# Muhammed Furkan YAGBASAN - 2099505
# 25.03.2019
# Ceng499 - THE1
#
####################################################
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import callbacks, optimizers 
import matplotlib.pyplot as plt
from PIL import Image

from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(4321)

Learning_Rate = 0.001

# initialize train set array
with open("dataset/train_set_label.txt") as fp:
	cnt = 0;
	for line in fp:
		cnt+=1;
num_of_training_exampes = cnt
train_set = np.zeros(shape=(num_of_training_exampes,128*64*3))
train_set_labels = np.zeros((num_of_training_exampes, 11))

# initialize test set array
with open("dataset/test_set_label.txt") as fp:
	cnt = 0;
	for line in fp:
		cnt+=1;
num_of_test_exampes = cnt
test_set = np.zeros(shape=(num_of_test_exampes,128*64*3))
test_set_labels = np.zeros((num_of_test_exampes,11))

# fill train set
with open("dataset/train_set_label.txt") as fp:
	cnt = 0
	for line in fp:
		words = line.strip().split(' ')
		image_path = words[0]
		image_label = int(words[1])

		im = Image.open("dataset/train_set/" + image_path)
		np_im = np.array(im)
		train_set[cnt] = np_im.flatten()
		train_set[cnt] *= 2/255.0  
		train_set[cnt] -= 1
		train_set_labels[cnt][image_label] = 1
		cnt+=1;

# fill test set
with open("dataset/test_set_label.txt") as fp:
	cnt = 0
	for line in fp:
		words = line.strip().split(' ')
		image_path = words[0]
		image_label = int(words[1])

		im = Image.open("dataset/test_set/" + image_path)
		np_im = np.array(im)
		test_set[cnt] = np_im.flatten()
		test_set[cnt] *= 2/255.0  
		test_set[cnt] -= 1
		test_set_labels[cnt][image_label] = 1
		cnt+=1;

# create layers of the model
inputs = Input(shape=(128*64*3,))
x = Dense(25, activation="sigmoid")(inputs)
x2 = Dense(25, activation="tanh")(x)
predictions = Dense(11, activation='softmax')(x2)

# arly stopping func, if val_loss does not improve for continous 5 epochs, stop
earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

# create model
model = Model(inputs=inputs, outputs=predictions)

# optimizer to arrange learning rate
opt_obj = optimizers.Adam(lr=Learning_Rate)

# compile the model
model.compile(optimizer=opt_obj,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# calculate test accuracy before training (randomized weights)
test_pred = model.predict(test_set)
test_acc = sum([np.argmax(test_set_labels[i])==np.argmax(test_pred[i]) for i in range(num_of_test_exampes)])/num_of_test_exampes
print("test accuracy before training >> %.2f" %(test_acc))

# train the model
history = model.fit(train_set, train_set_labels, batch_size=32, epochs=5000, validation_split=0.2, callbacks=[earlyStopping]) 

# test and traning accuracy after training is finished
test_pred = model.predict(test_set)
test_acc = sum([np.argmax(test_set_labels[i])==np.argmax(test_pred[i]) for i in range(num_of_test_exampes)])/num_of_test_exampes
training_acc=history.history["acc"][-1]
print("test accuracy after training >> %.2f" %(test_acc))
print("final training accuracy >> %.2f" %(training_acc))

# plot the training-validation loss graph
training_losses = history.history['loss']
validation_losses = history.history['val_loss']
epocs_number = range(1, len(training_losses)+1)
plt.plot(epocs_number, training_losses, label='Training Loss')
plt.plot(epocs_number, validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.title('Best Result for 3 Layers')
plt.legend()
plt.show()

# save the model
model.save('trained_model.h5')
