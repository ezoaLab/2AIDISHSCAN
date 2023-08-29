'''import os
import io
from PIL import Image
import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

    # Inference
    results = model(imgs, size=640)  # includes NMS
    return results

DETECTION_URL = "/v1/dish-detection/"
@app.route(DETECTION_URL, methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)

        results.save(save_dir='static')  # Save the result image in the 'static' folder

        # Pass the result image URL to the template
        result_image_url = url_for('static', filename='results0.jpg')
        return render_template('index.html', result_image_url=result_image_url)

    return render_template('index.html', result_image_url=None)  # Pass None if no result image yet

if __name__ == "__main__":
    # Specify the port you want to use (e.g., 8080)
    port = 46082
    # Run the Flask app with the specified port
    app.run(host='0.0.0.0', port=port)'''

'''import os
import io
from PIL import Image
import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join('results')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model.eval()

# Global variable to store the latest image filename
latest_image_filename = None

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

    # Inference
    results = model(imgs, size=640)  # includes NMS
    return results

DETECTION_URL = "/v1/dish-detection/"
@app.route(DETECTION_URL, methods=['GET', 'POST'])
def predict():
    global latest_image_filename  # Access the global variable

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)

        # Generate a unique filename for the image (you can use any method to ensure uniqueness)
        filename = f"result.jpg"
        save_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        results.save(save_path)

        # Update the latest image filename
        latest_image_filename = filename

        # Pass the result image URL to the template
        result_image_url = url_for('static', filename=latest_image_filename)
        return render_template('index.html', result_image_url=result_image_url)

    return render_template('index.html', result_image_url=latest_image_filename)

if __name__ == "__main__":
    # Specify the port you want to use (e.g., 8080)
    port = 46082
    # Run the Flask app with the specified port
    app.run(host='0.0.0.0', port=port)
'''

'''import os
import io
from PIL import Image
import torch
from flask import Flask, url_for, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = 'static'
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model.eval()

# Global variables to store the original image filename and the result image counter
original_filename = None
result_image_counter = 0

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

    # Inference
    results = model(imgs, size=640)  # includes NMS
    return results

DETECTION_URL = "/v1/dish-detection/"
@app.route(DETECTION_URL, methods=['GET', 'POST'])
def predict():
    global original_filename, result_image_counter

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)

        # Save the original filename of the uploaded image
        original_filename = file.filename

        # Generate a unique filename for the result image
        result_image_counter += 1
        exp_folder = f"exp{result_image_counter}"
        filename = f"{original_filename}_results.jpg"
        save_path = os.path.join('runs/detect', exp_folder, filename)
        results.save(save_path)

        # Pass the result image URL to the template
        result_image_url = url_for('runs', filename=os.path.join('detect', exp_folder, filename))
        return render_template('index.html', result_image_url=result_image_url)

    return render_template('index.html', result_image_url=None)

if __name__ == "__main__":
    # Specify the port you want to use (e.g., 8080)
    port = 46082
    # Run the Flask app with the specified port
    app.run(host='0.0.0.0', port=port)'''

'''import os
import io
from PIL import Image
import torch
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

RESULT_FOLDER = 'static'
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model.eval()

# Global variable to store the latest image filename
latest_image_filename = None

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

    # Inference
    results = model(imgs, size=640)  # includes NMS
    return results

DETECTION_URL = "/v1/dish-detection/"
@app.route(DETECTION_URL, methods=['GET', 'POST'])
def predict():
    global latest_image_filename  # Access the global variable

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)

        # Save the result image in the 'static' folder
        filename = "result.jpg"  # You can use any filename you want
        save_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        results.save(save_path)

        # Update the latest image filename
        latest_image_filename = filename
        print(latest_image_filename)

        # Redirect to the result page to display the result image
        return redirect(url_for('result'))

    return render_template('index.html', result_image_url=latest_image_filename)

@app.route('/result')
def result():
    global latest_image_filename

    # Pass the result image URL to the result.html template
    result_image_url = url_for('runs', filename=latest_image_filename)

    # Reset the latest_image_filename so that it doesn't show the result again after a refresh
    latest_image_filename = None

    return render_template('result.html', result_image_url=result_image_url)

if __name__ == "__main__":
    # Specify the port you want to use (e.g., 8080)
    port = 46082
    # Run the Flask app with the specified port
    app.run(host='0.0.0.0', port=port)'''

'''import os
from flask import Flask, request, redirect, url_for, render_template, jsonify
from PIL import Image
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load YOLOv5 model (Make sure you have the appropriate weights in the 'weights' folder)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model.eval()

def process_image(image_path):
    # Process image with YOLOv5
    results = model(image_path)
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image in the request.'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected.'}), 400
        
        if image_file:
            # Save the uploaded image to the 'uploads' folder
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)
            
            # Process the image with YOLOv5
            detections = process_image(image_path)
            
            # Return the results as JSON
            return jsonify({'image_path': image_path, 'detections': detections})
    
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True)'''


'''import argparse
import io
import os
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# force_reload = recache latest code
model = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model.eval()



@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        results = model(img)  # inference
        results.save("")
        results.render()  # updates results.ims with boxes and labels
        Image.fromarray(results.ims[0]).save("static/images/image0.jpg")
        #return redirect("static/images/image0.jpg")
        return print("success")

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    # app.run(host="0.0.0.0", port=args.port)
    app.run(host='0.0.0.0', debug=True, port=int(os.environ.get('PORT', 5000)))'''
