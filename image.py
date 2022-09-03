import numpy
import pandas
import PIL
from PIL import Image
import random
import torch
from torchvision import transforms
from IPython.display import clear_output

MODEL = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

HUE_DICT = {
'red':(255,0,0),
 'yellow': (255,255,0),
 'pink': (255,0,255),
 'green': (0,255,0),
 'orange': (255,128,0),
 'purple': (127,0,255),
 'blue': (0,0,255),
 'brown': (102,51,0),
 'grey': (128,128,128),
 'white': (255,255,255),
 'black': (0,0,0)}

class Image:
	def __init__(self, matrix: PIL.Image.Image):
		self.matrix = matrix
		self.size = matrix.size

	def genImage(self, name: str, hault_threshold: int =95, learning_rate: float = 0.1):
		k = 0
		place_count = 0
		item = name

		prev_image = self.matrix.copy()
		prev_image_value = dict(pred(prev_image))[item]

		while (prev_image_value < hault_threshold):
			clear_output(wait=True)

			self.matrix = prev_image.copy()
			self.placeRandCircle( HUE_DICT['red'])
			add_red_image_value = dict(pred(self.matrix))[item]
			place_count += 1
			if (add_red_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix.copy()
				print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_red_image_value}')
				prev_image_value = add_red_image_value
				k = k + 1
				continue

			self.matrix = prev_image.copy()
			self.placeRandCircle( HUE_DICT['yellow'])
			add_yellow_image_value = dict(pred(self.matrix))[item]
			place_count += 1
			if (add_yellow_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix.copy()
				print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_yellow_image_value}')
				prev_image_value = add_yellow_image_value
				k = k + 1
				continue

			self.matrix = prev_image.copy()
			self.placeRandCircle( HUE_DICT['pink'])
			add_pink_image_value = dict(pred(self.matrix))[item]
			place_count += 1
			if (add_pink_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix.copy()
				print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_pink_image_value}')
				prev_image_value = add_pink_image_value
				k = k + 1
				continue

			self.matrix = prev_image.copy()
			self.placeRandCircle( HUE_DICT['green'])
			add_green_image_value = dict(pred(self.matrix))[item]
			place_count += 1
			if (add_green_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix.copy()
				print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_green_image_value}')
				prev_image_value = add_green_image_value
				k = k + 1
				continue

			self.matrix = prev_image.copy()
			self.placeRandCircle( HUE_DICT['orange'])
			add_orange_image_value = dict(pred(self.matrix))[item]
			place_count += 1
			if (add_orange_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix.copy()
				print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_orange_image_value}')
				prev_image_value = add_orange_image_value
				k = k + 1
				continue

			self.matrix = prev_image.copy()
			self.placeRandCircle( HUE_DICT['purple'])
			add_purple_image_value = dict(pred(self.matrix))[item]
			place_count += 1
			if (add_purple_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix.copy()
				print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_purple_image_value}')
				prev_image_value = add_purple_image_value
				k = k + 1
				continue

			self.matrix = prev_image.copy()
			self.placeRandCircle( HUE_DICT['blue'])
			add_blue_image_value = dict(pred(self.matrix ))[item]
			place_count += 1
			if (add_blue_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix .copy()
				print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_blue_image_value}')
				prev_image_value = add_blue_image_value
				k = k + 1
				continue

			self.matrix = prev_image.copy()
			self.placeRandCircle( HUE_DICT['brown'])
			add_brown_image_value = dict(pred(self.matrix))[item]
			place_count += 1
			if (add_brown_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix.copy()
				print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_brown_image_value}')
				prev_image_value = add_brown_image_value
				k = k + 1
				continue

			self.matrix = prev_image.copy()
			self.placeRandCircle( HUE_DICT['grey'])
			add_grey_image_value = dict(pred(self.matrix))[item]
			place_count += 1
			if (add_grey_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix.copy()
				print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_grey_image_value}')
				prev_image_value = add_grey_image_value
				k = k + 1
				continue

			self.matrix = prev_image.copy()
			self.placeRandCircle( HUE_DICT['white'])
			add_white_image_value = dict(pred(self.matrix))[item]
			place_count += 1
			if (add_white_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix.copy()
				print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_white_image_value}')
				prev_image_value = add_white_image_value
				k = k + 1
				continue

			self.matrix = prev_image.copy()
			self.placeRandCircle(HUE_DICT['black'])
			add_black_image_value = dict(pred(self.matrix))[item]
			place_count += 1
			if (add_black_image_value > prev_image_value + learning_rate):
				prev_image = self.matrix.copy()
				prev_image_value = add_black_image_value
				k = k + 1
				continue
			print(f'Attempt No.{place_count} \nPlacement No.{k} \nprev: {prev_image_value} \ncurrent: {add_black_image_value}')

		print(f'Finished - After {k} changes')
		return ([pred(self.matrix), pred(prev_image), prev_image])

	def placeRandCircle(self, color: tuple, min_width: int = 20, max_width: int = 100):

		size_rand = random.randint(min_width, max_width)

		x_rand = random.randint(0, 255 - size_rand)
		y_rand = random.randint(0, 255 - size_rand)

		r_rand = random.randint(0, 255)
		g_rand = random.randint(0, 255)
		b_rand = random.randint(0, 255)

		from PIL import Image, ImageDraw
		draw = ImageDraw.Draw(self.matrix)

		draw.ellipse((x_rand, y_rand, x_rand + size_rand, y_rand + size_rand), fill=color)
		
	def placeRandPixel(self):
		
		x_rand = random.randint(0, 255)
		y_rand = random.randint(0, 255)

		r_rand = random.randint(0, 255)
		g_rand = random.randint(0, 255)
		b_rand = random.randint(0, 255)

		old_pixel_val = self.matrix.load()[x_rand,y_rand]
		new_pixel_val = (r_rand,g_rand,b_rand)

		self.matrix.load()[x_rand,y_rand] = (r_rand,g_rand,b_rand)

	def trick(self, name: str, learning_rate: float = 0, hault_threshold: int = 95):

		run_image = self.matrix.resize((256, 256))
		n_sub1_pred = pred(run_image)
		n_sub1_value = [item for item in n_sub1_pred if item[0] == name][0][1]
		k=0
		while ((n_sub1_pred[0][0] != name) or (n_sub1_pred[0][1]< hault_threshold)) and (n_sub1_value < hault_threshold):
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

			if current_value > n_sub1_value + learning_rate:
				print(f"change no. {k}")
				print(f"old n_sub1_value: {n_sub1_value}")
				n_sub1_pred = current_pred
				n_sub1_value = current_value

				print(f"new n_sub1_value: {n_sub1_value}")
				k=k+1
			else:
				run_image.load()[x_rand,y_rand] = old_pixel_val

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