from keras.models import Model, load_model
from PIL import Image
import numpy as np
import sys

if(len(sys.argv)!=2):
	print("usage: python runModel.py <model_file>")
	sys.exit(1)

# initialize test set array
with open("dataset/test_set_label.txt") as fp:
	cnt = 0;
	for line in fp:
		cnt+=1;
num_of_test_exampes = cnt
test_set = np.zeros(shape=(num_of_test_exampes,128*64*3))
test_set_labels = np.zeros((num_of_test_exampes,11))

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


model = load_model(sys.argv[1])

test_pred = model.predict(test_set)
test_acc = sum([np.argmax(test_set_labels[i])==np.argmax(test_pred[i]) for i in range(num_of_test_exampes)])/num_of_test_exampes

print("test accuracy after training >> %.2f" %(test_acc))
