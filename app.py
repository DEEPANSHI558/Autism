from flask import Flask, request, jsonify,render_template
import pickle
from PIL import Image
import numpy as np
import json
import os
import numpy as np
import tensorflow as tf
import base64
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename



# Register the 'FixedDropout' layer as a custom layer
tf.keras.utils.get_custom_objects().update({'FixedDropout': tf.keras.layers.Dropout})
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

# Load the model from an .h5 file
model = tf.keras.models.load_model('/home/deepanshi/practiceImage/best_model.h5')
model2=pickle.load(open('qqq.pkl','rb'))

UPLOAD_FOLDER='/home/deepanshi/practiceImage/static/images'

# Allowed file extensions
ALLOWED_EXTENSIONS=set(['txt', 'jpg', 'png','pdf','jpeg','gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Function to save the file to the server
def save_file(file):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return full_path

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/image')
def image():
    return render_template('image.html')

# for image classification using EfficientNet B) that gave 81% accuracy
@app.route('/predict', methods=['POST'])
def predict():
    # Load and preprocess the image
    image = request.files['file']
    if image and allowed_file(image.filename):
         filename = save_file(image)
         print(filename)
    else:
            return jsonify({'msg': 'Invalid file type'})

    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    # Make a prediction using the model
    predictions = model.predict(img)

    # Return the predicted class as a JSON object
    # return jsonify({'class': int(np.argmax(predictions))})
    return render_template('imageafter.html',data=int(np.argmax(predictions)),image_file=image)
    

# for Questionnaire based prediction 
@app.route('/predict/questionnaire',methods=['POST'])
def qprediction():
    data1=request.form['q1']
    data2=request.form['q2']
    data3=request.form['q3']
    data4=request.form['q4']
    data5=request.form['q5']
    data6=request.form['q6']    
    data7=request.form['q7']
    data8=request.form['q8']
    data9=request.form['q9']
    data10=request.form['q10']
    arr=[[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]]
    numeric_array = np.array(arr, dtype=np.int64)
    pred=model2.predict(numeric_array)
    ans=pred[0].tolist()
    json_data=json.dumps(ans)
    return render_template('after.html',data=ans)



if __name__ == '__main__':
   app.run(debug=True,port=5000)
