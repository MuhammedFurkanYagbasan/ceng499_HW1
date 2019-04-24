trainer.py -> the source code (produces a model file named trained_model.h5)
testExistingModel -> usage: python testExistingModel <modelfile>
report.pdf -> report of the homework

exampleModel.h5 -> trained model with:
	SGD optimizer with 0.00003 learning rate
	2 hidden layers with 25 nodes each
	last around 3000 epochs
	can run with (python testExistingModel exampleModel.h5)
	gives 0.71 accuracy on test test


dataset folder that downloaded from "http://user.ceng.metu.edu.tr/~artun/ceng499/dataset.zip" should be placed in the same folder with sorce code. (number of examples in data set can be changed without changing the source code.)
