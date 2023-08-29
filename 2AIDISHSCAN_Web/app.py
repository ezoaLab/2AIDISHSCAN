
import io
import os
import json
from PIL import Image

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect,session
from flask_session import Session
app = Flask(__name__)

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model_dish = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
model_calorie = torch.hub.load('yolov5', 'custom', path='models/calorie.pt', source='local') 
model_dish.eval()
model_calorie.eval()

def get_prediction(img_bytes, model):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

    # Inference
    results = model(imgs, size=640)  # includes NMS

    return results



@app.route("/", methods=['GET', 'POST'])
def predict():

    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()

        # Get the selected action from the form
        selected_action = request.form.get('action')

        if selected_action == 'dish':
            model = model_dish
        elif selected_action == 'calorie':
            model = model_calorie
        else:
            return render_template('index.html', error_message="Invalid action selected!")

        results = get_prediction(img_bytes, model)
        results.render()

        Image.fromarray(results.ims[0]).save("static/result.jpg")

        #return redirect("static/result.jpg")
        return render_template('result.html', result_image="static/result.jpg")
        
        
    return render_template('index.html')

if __name__ == "__main__":
    # Specify the port you want to use (e.g., 8080)
    port = 46082
    # Run the Flask app with the specified port
    app.run(host='0.0.0.0', port=port, debug=True)

