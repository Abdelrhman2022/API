# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 00:57:33 2022

@author: Abdelrahman Ragab
"""
import keras
import efficientnet.tfkeras as efn  
import numpy as np
import cv2
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

 
app = Flask(__name__)
 
app.secret_key = "caircocoders-ednalan"
 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def load_ben_color(image, sigmaX=10):
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image


def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def get_predictions(img_path):
    
     image = keras.preprocessing.image.load_img(img_path, target_size=(224, 224, 1))
     image_array = keras.preprocessing.image.img_to_array(image)
     img = crop_image_from_gray(image_array)
     img = load_ben_color(img)
     img = img.reshape(1,224,224,3)
     img = np.float32(img) / 255.0
     model = keras.models.load_model(r'A:\FCAI-HU\ML\API\FinalRetinopathyModel.h5')
      
     pred = model.predict(img)
     return np.argmax(pred)
 
def get_className(classNo):
    """
    Input: pridection result 0-> Mask Found, 1-> Mask Not Found
    Return Text Message
    """
    if classNo==0:
        return " Mask is Up"
    elif classNo==1:
        return " Warning No Mask!"



@app.route("/shutdown", methods=['GET'])
def shutdown():
    print("hello")
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running werkzeug')
    shutdown_func()
    return "Shutting down..."    
 
@app.route('/')
def main():
    return 'Hellow world'
 
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
        classNo = get_predictions(file_path)
        success = True
    else:
        errors[file.filename] = "File type isn't allowed"
 

    if success:
        resp = jsonify({'message' : 'Files successfully uploaded', 'Class' : str(classNo) })

        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
 
if __name__ == '__main__':
    app.run(debug=True)