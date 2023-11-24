'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			GG_3248
# Author List:		Chirag Mahajan, Shridhar Kamat
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]


import pandas 
import torch
import numpy 

input_size=0
output_size=0

def data_preprocessing(task_1a_dataframe):
	encoded_dataframe = pandas.get_dummies(task_1a_dataframe, columns=['Education', 'City', 'Gender', 'EverBenched'])
	encoded_dataframe = encoded_dataframe.fillna(encoded_dataframe.mean())
	return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
	features = encoded_dataframe.drop(columns=['LeaveOrNot'])  
	target = encoded_dataframe['LeaveOrNot']
	features_and_targets = [features, target]
	global input_size
	global output_size
	input_size =  len(features.columns)
	output_size = 1
	return features_and_targets


def load_as_tensors(features_and_targets):
	features, target = features_and_targets
	X_np = features.values.astype(numpy.float32)
	y_np = target.values.astype(numpy.float32)
	X = torch.tensor(X_np, dtype=torch.float32)	
	y = torch.tensor(y_np, dtype=torch.float32)
	dataset = torch.utils.data.TensorDataset(X, y)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=72, shuffle=True)

	tensors_and_iterable_training_data = [X, y, dataloader]
	return tensors_and_iterable_training_data

class Salary_Predictor(torch.nn.Module):
	def __init__(self):
		super(Salary_Predictor, self).__init__()
		self.fc1 = torch.nn.Linear(input_size,72)
		self.fc2 = torch.nn.Linear(72, 64)
		self.fc3 = torch.nn.Linear(64, 1)

	def forward(self, x):
		x = torch.nn.functional.relu(self.fc1(x))
		x = torch.nn.functional.relu(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		predicted_output = x
		return predicted_output

def model_loss_function():
	loss_function = torch.nn.BCELoss()
	return loss_function

def model_optimizer(model):
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	return optimizer

def model_number_of_epochs():
	number_of_epochs = 100
	return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
	for epoch in range(number_of_epochs):
		for batch in tensors_and_iterable_training_data[2]:
			inputs, labels = batch
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = loss_function(outputs, labels.unsqueeze(1))
			loss.backward()
			optimizer.step()

	return model

def validation_function(trained_model, tensors_and_iterable_training_data):
	X_val_tensor, y_val_tensor, _ = tensors_and_iterable_training_data
	with torch.no_grad():
		val_predictions = trained_model(X_val_tensor)
		val_predictions_binary = (val_predictions >= 0.5).float()  


	correct_predictions = (val_predictions_binary == y_val_tensor.unsqueeze(1)).sum().item()
	total_samples = y_val_tensor.size(0)
	model_accuracy = correct_predictions / total_samples
	return model_accuracy

if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	
	# model is an instance of the class that defines the architecture of the model
	model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()
	
	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
					loss_function, optimizer)
	
	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")