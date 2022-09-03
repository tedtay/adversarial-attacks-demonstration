import numpy
import pandas
import PIL
from PIL import Image
import random
import torch
from torchvision import transforms
from IPython.display import clear_output

MODEL = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

class Image:
	
	def __init__(self, matrix: PIL.Image.Image):
		self.matrix = matrix

	def trick(self, name: str, hault_threshold: int = 95):

		run_image = self.matrix.resize((256, 256))
		n_sub1_pred = pred(run_image)
		n_sub1_value = [item for item in n_sub1_pred if item[0] == name][0][1]
		k=0
		while (n_sub1_pred[0][0] != name) or (n_sub1_pred[0][1] < hault_threshold):
			clear_output(wait=True)

			x_rand = random.randint(0, 255)
			y_rand = random.randint(0, 255)

			r_rand = random.randint(0, 255)
			g_rand = random.randint(0, 255)
			b_rand = random.randint(0, 255)

			old_pixel_val = run_image.load()[x_rand,y_rand]
			new_pixel_val = (r_rand,g_rand,b_rand)

			run_image.load()[x_rand,y_rand] = (r_rand,g_rand,b_rand)
			current_pred = pred(run_image)
			current_value = [item for item in current_pred if item[0] == name][0][1]

			if current_value > n_sub1_value:
				print(f"change no. {k}")
				print(f"old n_sub1_value: {n_sub1_value}")
				#print(f"BETTER ({x_rand},{y_rand}) old: {old_pixel_val}  new: {new_pixel_val}")
				n_sub1_pred = current_pred
				n_sub1_value = current_value

				print(f"new n_sub1_value: {n_sub1_value}")
				k=k+1
			else:
				run_image.load()[x_rand,y_rand] = old_pixel_val
				#print(f"WORSE ({x_rand},{y_rand}) old: {old_pixel_val}  new: {new_pixel_val}")

		print(f'Finished - After {k} changes')
		return ([pred(self.matrix.resize((256, 256))),pred(run_image),run_image])
	

def preprocessImage(img_param: PIL.Image.Image) -> torch.Tensor:
	# Create a preprocessing pipeline
	#
	preprocess = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)])
	#
	# Pass the image for preprocessing and the image preprocessed
	#

	return preprocess(img_param)

def pred(img_param: PIL.Image.Image, param_amount: int = 1000) -> list:
	img_tensor = preprocessImage(img_param)
	#
	# Reshape, crop, and normalize the input tensor for feeding into network for evaluation
	#
	batch_img_cat_tensor = torch.unsqueeze(img_tensor, 0)

	MODEL.eval()
	#
	# Get the predictions of image as scores related to how the loaded image
	# matches with 1000 ImageNet classes. The variable, out is a vector of 1000 scores
	#
	out = MODEL(batch_img_cat_tensor)

	# Load the file containing the 1,000 labels for the ImageNet dataset classes
	#
	with open('/Users/ted.taylor/Downloads/imagenet_classes.txt') as f:
		labels = [line.strip() for line in f.readlines()]
	#
	# Find the index (tensor) corresponding to the maximum score in the out tensor.
	# Torch.max function can be used to find the information
	#
	_, index = torch.max(out, 1)
	#
	# Find the score in terms of percentage by using torch.nn.functional.softmax function
	# which normalizes the output to range [0,1] and multiplying by 100
	#
	percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
	#
	# Print the name along with score of the object identified by the model
	#
	#print(labels[index[0]], percentage[index[0]].item())
	#
	# Print the top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores.
	#
	_, indices = torch.sort(out, descending=True)
	preds = [(labels[idx], percentage[idx].item()) for idx in indices[0][:param_amount]]

	return (preds)