import gunicorn
import keras
import efficientnet.tfkeras as efn
import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

 
app = Flask(__name__)
 
app.secret_key = "caircocoders-ednalan"
 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

model = keras.models.load_model(r'.\FinalRetinopathyModel.h5')
 
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
    
     image = keras.preprocessing.image.load_img(img_path, target_size=(224, 224, 3))
     image_array = keras.preprocessing.image.img_to_array(image)
     
     img = crop_image_from_gray(image_array)
     img = load_ben_color(img)

     image = tf.keras.preprocessing.image.smart_resize(img, (224,224))
     img = image.reshape(1,224,224,3)
     img = np.float32(img) / 255.0
     pred = model.predict(img)
     return np.argmax(pred)
 
def get_className(classNo):
    """
    Input: pridection result 0-> Normal , 1-> Mild
    2-> Modrate, 3-> severe, 4-> Proliferative 
    """
    if classNo==0:
        return "Normal (No DR)"
    elif classNo==1:
        return "Mild"
    elif classNo==2:
        return "Modrate"
    elif classNo==3:
        return "severe"
    elif classNo==4:
        return "Proliferative DR"


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
    return 'Hello world'


    
@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part

    if 'imagefile' not in request.files:
        resp = jsonify(
            {'message' : 'No file exists',
             'id' : '-1'
             })
        return resp
            
    file = request.files['imagefile']
    errors = {}
    success = False
    classNo = None
    className = None
    file_path = None
          
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        classNo = get_predictions(file_path)
        className = get_className(classNo)
        success = True
        #os.remove(file_path)
    else:
        errors["message"] = "File type isn't allowed"
        errors["id"] = "-2"
 

    if success:
        resp = jsonify(
            {'message' : 'Image successfully uploaded',
             'ClassNo' : str(classNo),
             "ClassName" : className
             })
        

        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    
    @app.after_request
    def delete(resp):
        os.remove(file_path)
 
if __name__ == '__main__':
    app.run()


