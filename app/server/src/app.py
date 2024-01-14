from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os

# Check if the code is running on Render
on_render = os.environ.get('RENDER', False)

# If running on Render, use CPU only
if on_render:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    # If running locally, use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename
from werkzeug.wrappers import Request
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "broccoli"

app.config["MODEL"] = load_model(os.path.join("model", "weatherclass.h5"))
app.config["UPLOAD_FOLDER"] = "user_uploads"
app.config["ALLOWED_EXT"] = {"png", "jpg", "jpeg", "gif", "bmp"}

# create upload dir if it does not exist
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

@app.route('/')
def home(message = "", img=None):
    session_remove_img = session.get("remove_img", True)
    
    if session_remove_img:
        clear_img_data()
        return render_template('index.html', img=None)
    
    session_msg = session.get("msg", message)    
    session_pred_res = session.get("pred_result", "")    
    session_pred_conf = session.get("pred_confidence", "")    
    session_img = session.get("img_info", img)
        
    if session_img:
        session_img = session_img["filename"] 
        
    session["remove_img"] = True
    
    print("results: ", session_pred_res, " conf: ", session_pred_conf)
    return render_template('index.html', message=session_msg, img=session_img, pred_result=session_pred_res, pred_conf=session_pred_conf)
        
@app.route('/upload', methods=['POST'])
def upload():
    clear_img_data()
        
    uploaded_file = request.files['file']
    base_name, ext = os.path.splitext(secure_filename(uploaded_file.filename))

    if ext.lower() == '.jpg':
        new_filename = base_name + '.jpeg'
        uploaded_file.filename = secure_filename(new_filename)
    elif not (ext.lower() == ".jpeg" or ext.lower() == ".gif" or ext.lower() == ".png" or ext.lower() == ".bmp"):
        return render_template('index.html', message="Only PNG, JPG, JPEG, BMP, and GIF files are supported")
    
    # temporarily save the file to the system
    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(file_path)

    # store img info in session variable
    session["img_info"] = {"file_path": file_path, "filename": filename}

    # read image using OpenCV
    img = cv2.imread(file_path)

    # resize the image to have dimensions 256 x 256
    resized = tf.image.resize(img, (256, 256))
    
    # predict weather using normalized data
    yhat = app.config["MODEL"].predict(np.expand_dims(resized/255, 0))
    
    if yhat < 0.5:
        result = "rainy"
    else:
        result = "snowy"
    
    session["pred_confidence"] = int((abs(yhat - 0.5) + 0.5) * 100)
    session["pred_result"] = result    
    session["remove_img"] = False  
    
    return redirect(url_for('home'))

@app.route('/user_uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('user_uploads', filename)

def clear_img_data():
    img_info = session.pop('img_info', None)

    # Delete the previously uploaded image from the server
    try:
        if img_info:
            print()
            os.remove(img_info['file_path'])
    except Exception as e:
        print("file not found")
        
    session.clear()  

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
