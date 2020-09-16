# Libraries
import io
import string
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
import numpy

from torchvision import models
from flask import Flask, jsonify, request, render_template
from PIL import Image
from networks_binary.vgg16_ft import vgg16_ft
from networks_categories.vgg16_1_one_more_linear_layer import vgg16_1_one_more_linear_layer
from networks_categories.utils import masked_softmax

app = Flask(__name__)

class_index_binary = {"0": "Non-skin", "1": "Skin"}
class_index_categories = json.load(open('./categories.json'))

# Binary
model_binary = vgg16_ft()
best_path = './networks_binary/best1.tar'
best_checkpoint = torch.load(best_path, map_location=torch.device('cpu'))
model_binary.load_state_dict(best_checkpoint['net_state_dict'])
model_binary.eval()

# Categories
model_categories = vgg16_1_one_more_linear_layer()
best_path = './networks_categories/best1.tar'
best_checkpoint = torch.load(best_path, map_location=torch.device('cpu'))
model_categories.load_state_dict(best_checkpoint['net_state_dict'])
model_categories.eval()

size=224
mean = [0.587, 0.448, 0.417]
std = [0.262, 0.227, 0.223]
def transform_image(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(size),
                              transforms.CenterCrop((size,size)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean,std)])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)

def get_prediction_binary(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	outputs = model_binary.forward(tensor) # this are all probs
	all_probs = list(outputs.data.numpy().flatten())
	prob, pred = torch.max(outputs, 1)  # this is the result prob and result pred just 1
	pred_idx = str(pred.item())
	return pred_idx, all_probs

def get_prediction_categories(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	outputs = model_categories.forward(tensor) # this are all probs
	outputs = masked_softmax(vector=outputs, mask=[1, 50, 69, 73, 139, 222, 235, 253, 257, 296], dim=1) # categories to mask out
	#outputs = torch.nn.functional.softmax(outputs,dim=1)  # this are probs normalized
	all_probs = list(outputs.data.numpy().flatten())
	prob, pred = torch.max(outputs, 1)  # this is the result prob and result pred just 1
	pred_idx = str(pred.item())
	return pred_idx, all_probs

# Treat the web process
@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			return redirect(request.url)
		file = request.files.get('file')
		if not file:
			return
		img_bytes = file.read()
		pred_idx_binary, _ = get_prediction_binary(img_bytes)
		pred_idx_categories, all_probs_categories = get_prediction_categories(img_bytes)
		# Get the probable diseases the ones not zero
		probable_diseases = {}
		for i in range(len(all_probs_categories)):
			value = all_probs_categories[i]
			if value != 0:
				key = str(class_index_categories[str(i)])
				probable_diseases[key] = value
		probable_diseases = {k: v for k, v in sorted(probable_diseases.items(), key=lambda item: item[1], reverse=True)}
		# To print a table in html
		Table = []
		for key, value in probable_diseases.items():    # or .items() in Python 3
		    temp = []
		    temp.extend([key,value])  #Note that this will change depending on the structure of your dictionary
		    Table.append(temp)

		return render_template('result.html',
		 						result_binary = class_index_binary[pred_idx_binary],
								result_categories = class_index_categories[pred_idx_categories],
								table=Table)

	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)
