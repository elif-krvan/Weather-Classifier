from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
import numpy as np
import base64
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

@app.route('/')
def home(message = "", img=None):
    session_msg = session.get("msg", message)    
    session_img = session.get("img_info", img)
    
    if session_img:
        session_img = session_img["filename"] 
     
    session.clear()     
        
    print("ses image: ",session_img)
    return render_template('index.html', message=session_msg, img=session_img)
        
@app.route('/upload', methods=['POST'])
def upload():
    img_info = session.pop('img_info', None)

    # Delete the previously uploaded image from the server
    try:
        if img_info:
            os.remove(img_info['file_path'])
    except Exception as e:
        print("file not found")
        
    uploaded_file = request.files['file']
    
    base_name, ext = os.path.splitext(secure_filename(uploaded_file.filename))
    
    if ext.lower() == '.jpg':
        new_filename = base_name + '.jpeg'
        uploaded_file.filename = secure_filename(new_filename)
    elif not (ext.lower() == "jpg" or ext.lower() == "gif" or ext.lower() == "png" or ext.lower() == "bmp"):
        return render_template('index.html', message="Only PNG, JPG, JPEG, BMP, and GIF files are supported")
    
    print(uploaded_file.filename)
    
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
        msg = "Image is rain"
    else:
        msg = "Image is snow"
    
    session['msg'] = msg
    
    # # temporarily save the file to the system
    # uploaded_file = request.files['file']
    # print("before save")
    # print(uploaded_file)
    # print(uploaded_file.filename)
    # filename = secure_filename(uploaded_file.filename)
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # uploaded_file.save(file_path)
    
    # # store img info in session variable
    # session["img_info"] = {"file_path": file_path, "filename": filename}
    
    return redirect(url_for('home'))

@app.route('/user_uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('user_uploads', filename)

# @app.route('/clear', methods=['POST'])
# def clear():
#     print("clear eyle")
#     session.clear
#     return a

if __name__ == '__main__':
    app.run(debug=True)
