# -*- coding: utf-8 -*-
"""


@author: Abdelrahman Ragab
"""

from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from keras.models import load_model
 
app = Flask(__name__)
 
app.secret_key = "caircocoders-ednalan"
 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def preprocessing(img):
    """
    Input: Image with RGB scale
    Return: Image which enhanced with equalized Histogram
    """
    img=img.astype("uint8")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img/255
    return img

def faceMaskDetection(image_path):
    model = load_model(r'C:\Users\Abdelrahman Ragab\Downloads\Compressed\Project\FaceMaskDetection.h5')
    image = np.empty((1,224,224,3))
 
    # Read the frame
    image = cv2.imread(image_path)
    # Capture facees 

    
    # Find coordinates of face from whole image
    img=cv2.resize(image, (224,224),3)
    img=preprocessing(img)
    img=img.reshape(1,224,224,3)
    
    # Predict model 
    prediction=model.predict(img)
    classIndex=np.argmax(prediction)
    
    return classIndex

def get_className(classNo):
    """
    Input: pridection result 0-> Mask Found, 1-> Mask Not Found
    Return Text Message
    """
    if classNo==0:
        return " Mask is Up"
    elif classNo==1:
        return " Warning No Mask!"
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route("/shutdown", methods=['GET'])
def shutdown():
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running werkzeug')
    shutdown_func()
    return "Shutting down..."    
 
@app.route('/')
def main():
    return 'Homepage'
 
@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part

 
    file = request.files['imagefile']
    errors = {}
    success = False
    classNo = None
    
          
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        classNo = faceMaskDetection(file_path)
        success = True
    else:
        errors[file.filename] = "File type isn't allowed"
 
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded', 'Class' : str(classNo) })
        # resp = jsonify({'message' : 'Files successfully uploaded'})

        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
 
if __name__ == '__main__':
    app.run(debug=True)