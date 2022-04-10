
import gunicorn
import numpy as np
import cv2
from flask import Flask, request, render_template 
import os
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config[‘MAX_CONTENT_PATH’] = # in Byte




@app.route("/",  methods=['GET'])
def index():
    return render_template("index.html")  
    
@app.route("/", methods=['POST'])
def home():

    return render_template("index.html") 







if __name__ == "__main__":
  app.run()
  
  