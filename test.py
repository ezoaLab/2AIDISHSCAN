'''from flask import Flask, request, send_file
import os

from datetime import datetime
from io import BytesIO
from PIL import Image
import base64
import torch
import cv2
import numpy as np

app = Flask(__name__)
last_uploaded_image = None
upload_counter = 0

# Load YOLOv5 model

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # for file/URI/PIL/cv2/np inputs and NMS
def process_image(image ):
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
            #base64_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        
        return "good"

    except Exception as e:
        return "bad "
@app.route('/', methods=['POST'])
def upload_image():
    

    global last_uploaded_image, upload_counter

    if 'image' not in request.files:
        return 'No file part', 400

    image = request.files['image']
    if image.filename == '':
        return 'No selected file', 400
    


    upload_counter += 1
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    new_filename = f'image_{timestamp}_{upload_counter}.jpg'
    process_image(image )
    
    image.save(os.path.join('uploads', new_filename))
    last_uploaded_image = new_filename
    return 'Image saved on server', 200

@app.route('/get_last_image', methods=['GET'])
def get_last_image():
    global last_uploaded_image

    if last_uploaded_image is None:
        return 'No image uploaded', 404

    image_path = os.path.join('uploads', last_uploaded_image)
    return send_file(image_path, mimetype='image/jpg')
    print(last_uploaded_image)'''
'''files = os.listdir('uploads')
    
    if not files:
        return 'No image uploaded', 404

    first_uploaded_image = files[0]
    image_path = os.path.join('uploads', first_uploaded_image)
    return send_file(image_path, mimetype='image/jpg')'''


from flask import Flask, request, send_file
import os

from datetime import datetime
from io import BytesIO
from PIL import Image
import torch
import cv2
import base64
import numpy as np

app = Flask(__name__)
last_uploaded_image = None
upload_counter = 0

# Load YOLOv5 model
#model_1 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # for file/URI/PIL/cv2/np inputs and NMS

#model_2 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


dish = torch.hub.load('yolov5', 'custom', path='models/dish.pt', source='local')
calorie = torch.hub.load('yolov5', 'custom', path='models/calorie.pt', source='local') 

def process_image(image_path, model):
    try:
        image_cv2 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)#import for keeping the color same 

        # Perform inference using the specified YOLOv5 model
        results = model(image_cv2)
        results.render()

        base64_images = []
        for im in results.ims:
            buffered = BytesIO()
            im_base64 = Image.fromarray(im)
            im_base64.save(buffered, format="JPEG")
            base64_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))

        return base64_images

    except Exception as e:
        return []

# ...
@app.route('/', methods=['POST'])
def upload_image():
    global last_uploaded_image, upload_counter

    if 'image' not in request.files:
        return 'No file part', 400

    image = request.files['image']
    if image.filename == '':
        return 'No selected file', 400

    upload_counter += 1
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    new_filename = f'image_{timestamp}_{upload_counter}.jpg'

    image.save(os.path.join('uploads', new_filename))

   

    # Determine the model to use based on user's choice
    model_name = request.form.get('model_name')  # You need to pass the selected model name from the Flutter app
    if model_name == 'dish':
        current_model = dish
    elif model_name == 'calorie':
        current_model = calorie
    else:
        return 'Invalid model selection', 400
    
    # Process the image and get the result

    processing_result = process_image(os.path.join('uploads', new_filename),current_model)

    if processing_result:
        # Save processed image(s)
        for idx, base64_image in enumerate(processing_result):
            processed_filename = f'processed_{new_filename[:-4]}_{idx}.jpg'
            processed_image_path = os.path.join('proceed', processed_filename)
            with open(processed_image_path, 'wb') as f:
                f.write(base64.b64decode(base64_image.encode('utf-8')))

        last_uploaded_image = processed_filename
        return 'Image saved on server', 200
    else:
        return 'Image processing failed', 500

    
    

@app.route('/get_last_image', methods=['GET'])
def get_last_image():
    global last_uploaded_image

    if last_uploaded_image is None:
        return 'No image uploaded', 404

    image_path = os.path.join('proceed', last_uploaded_image)
    return send_file(image_path, mimetype='image/jpg')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('proceed'):
        os.makedirs('proceed')
    app.run(host='0.0.0.0', port=5000)




