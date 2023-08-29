'''from flask import Flask, render_template


#create a flask instance
app= Flask(__name__)


# create a route
@app.route('/')

def index():
    #return "<h1>Hello Ezoa!</>"
    return render_template('index.html')

@app.route('/user/<name>')

def user(name):
    #return "<h1> Hello {}</h1>".format(name)
    return render_template("user.html", user_name=name)'''

'''bon from flask import Flask, request, jsonify
import base64 
import torch
app = Flask(__name__)
model_dish = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model_calorie = torch.hub.load('yolov5', 'custom', path='models/calorie.pt', source='local') 

@app.route('/upload', methods=['POST'])
def upload_image():
    image_data = request.form.get('image')

    image_bytes = base64.b64decode(image_data)
    
    # Your processing logic here (e.g., save the image, perform some operations)

    # Send the processed image back to the Flutter app
    return jsonify({'image': base64.b64encode(image_bytes).decode('utf-8')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    bon 
    
    '''

'''from flask import Flask, request, jsonify
import base64
import torch

app = Flask(__name__)
model_dish = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model_calorie = torch.hub.load('yolov5', 'custom', path='models/calorie.pt', source='local') '''
'''
@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json  # Assuming the request contains a JSON payload
    image_data = data.get('image')
    model = data.get('model')

    # Decode the base64 image data
    image_bytes = base64.b64decode(image_data)

    # Convert the image bytes to a PyTorch tensor
    image_tensor = torch.tensor([image_bytes])

    # Load the appropriate model based on the request
    if model == 'dish':
        model = model_dish
    elif model == 'calorie':
        model = model_calorie
    else:
        return jsonify({'error': 'Invalid model selection'})

    # Process the image using the selected model
    results = model(image_tensor)

    # Process the results and get relevant information about the dish or calorie
    # Replace this part with the specific processing logic for your models

    # For example, you might retrieve the labels, confidence scores, bounding boxes, etc.
    predictions = get_predictions(results)

    # Return the result to the Flutter app as a JSON response
    return jsonify({'predictions': predictions})'''

'''def get_predictions(results):
    # Replace this function with the specific processing logic to extract relevant information from the model results
    # For example, if 'results' is a list of objects, you might extract the label and confidence score for each object.
    # You can then format this information into a dictionary or list of dictionaries to be sent back as a response.
    # Example:
    # predictions = [{'label': 'dish', 'confidence': 0.85}, {'label': 'plate', 'confidence': 0.92}]
    # return predictions

    # Assuming 'results' is a list of tensor(s) containing bounding boxes and labels

    predictions = []
    for result in results:
        # Process each bounding box and label in the result tensor
        for box_info in result:
            # Extract information from the result tensor
            label = box_info[5]  # Assuming the label is at index 5
            confidence = box_info[4]  # Assuming the confidence score is at index 4

            # Create a dictionary to represent the prediction
            prediction = {'label': label.item(), 'confidence': confidence.item()}

            # Add the prediction to the list
            predictions.append(prediction)

    return predictions

import numpy as np

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json  # Assuming the request contains a JSON payload
    image_data = data.get('image')

    model = data.get('model')

    # Decode the base64 image data
    image_bytes = base64.b64decode(image_data)

    # Convert the image bytes to a NumPy array
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)

    # Create a PyTorch tensor from the NumPy array
    image_tensor = torch.tensor(image_np)

    # Add a batch dimension to the image tensor
    image_tensor = image_tensor.unsqueeze(0)

    # Load the appropriate model based on the request
    if model == 'dish':
        model = model_dish
    elif model == 'calorie':
        model = model_calorie
    else:
        return jsonify({'error': 'Invalid model selection'})

    # Process the image using the selected model
    results = model(image_tensor)

    # Process the results and get relevant information about the dish or calorie
    # Replace this part with the specific processing logic for your models

    # For example, you might retrieve the labels, confidence scores, bounding boxes, etc.
    predictions = get_predictions(results)

    # Return the result to the Flutter app as a JSON response
    return jsonify({'predictions': predictions})





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)'''


'''import base64
import cv2
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the YOLOv5 models
model_dish = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model_calorie = torch.hub.load('yolov5', 'custom', path='models/calorie.pt', source='local') 


model_dish.eval()
model_calorie.eval()

def perform_inference(image_bytes, model_name):
    # Perform inference using the specified YOLOv5 model
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if model_name == 'dish':
        results = model_dish(img)
    elif model_name == 'calorie':
        results = model_calorie(img)
    else:
        return []

    # Process inference results
    inference_results = []
    for result in results.pred[0]:
        class_name = result.names[int(result[5])]
        confidence = float(result[4])
        inference_results.append({'class': class_name, 'confidence': confidence})

    return inference_results

@app.route('/inference', methods=['POST'])
def inference():
    data = request.form
    image_data = data.get('image')
    model_name = data.get('action')
    
    image_bytes = base64.b64decode(image_data)
    
    inference_results = perform_inference(image_bytes, model_name)
    
    return jsonify({'results': inference_results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)'''


'''from flask import Flask, request, jsonify
import base64
import torch
import numpy as np
from PIL import Image

app = Flask(__name__)
model_dish = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model_calorie = torch.hub.load('yolov5', 'custom', path='models/calorie.pt', source='local')

def preprocess_image(image_data):
    # Decode the base64 image data
    image_bytes = base64.b64decode(image_data)

    # Convert the image bytes to a NumPy array
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)

    # Create a PIL image from the NumPy array
    image_pil = Image.fromarray(image_np)

    # Resize the image to the expected input size of the YOLOv5 model (e.g., 640x640)
    input_size = (640, 640)
    image_pil = image_pil.resize(input_size, Image.BICUBIC)

    # Convert the PIL image to a NumPy array
    image_np = np.array(image_pil)

    # Check if the image is grayscale (single channel)
    if len(image_np.shape) == 2:
        # Convert grayscale image to RGB format
        image_np = np.stack([image_np] * 3, axis=-1)

    # Check if the image has an alpha channel (RGBA format)
    if image_np.shape[2] == 4:
        # Remove alpha channel to get RGB format
        image_np = image_np[:, :, :3]

    # Normalize the image data to [0, 1] range
    image_np = image_np / 255.0

    # Transpose the image array to match the YOLOv5 model's input format (C, H, W)
    image_np = np.transpose(image_np, (2, 0, 1))

    # Add a batch dimension to the image tensor
    image_np = np.expand_dims(image_np, axis=0)

    # Create a PyTorch tensor from the NumPy array
    image_tensor = torch.tensor(image_np, dtype=torch.float32)

    return image_tensor


def get_predictions(results):
    # Assuming 'results' is a list of tensor(s) containing bounding boxes and labels

    predictions = []
    for result in results:
        # Process each bounding box and label in the result tensor
        for box_info in result:
            # Extract information from the result tensor
            label = int(box_info[5])  # Assuming the label is at index 5
            confidence = float(box_info[4])  # Assuming the confidence score is at index 4

            # Create a dictionary to represent the prediction
            prediction = {'label': label, 'confidence': confidence}

            # Add the prediction to the list
            predictions.append(prediction)

    return predictions

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json  # Assuming the request contains a JSON payload
    image_data = data.get('image')
    model = data.get('model')

    # Preprocess the image
    image_tensor = preprocess_image(image_data)

    # Load the appropriate model based on the request
    if model == 'dish':
        model = model_dish
    elif model == 'calorie':
        model = model_calorie
    else:
        return jsonify({'error': 'Invalid model selection'})

    # Process the image using the selected model
    results = model(image_tensor)

    # Process the results and get relevant information about the dish or calorie
    predictions = get_predictions(results)

    # Return the result to the Flutter app as a JSON response
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)'''

from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import base64
import torch
import cv2
import numpy as np



app = Flask(__name__)

# Load YOLOv5 model

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # for file/URI/PIL/cv2/np inputs and NMS

@app.route('/', methods=['POST'])
def process_image():
    try:
        image_data = request.files['image'].read()
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        # Perform inference using YOLOv5
        results = model(image)
        results.render()
        
        base64_images = []
        for im in results.ims:
            buffered = BytesIO()
            im_base64 = Image.fromarray(im)
            im_base64.save(buffered, format="JPEG")
            base64_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        
        return jsonify({'images': base64_images})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)



