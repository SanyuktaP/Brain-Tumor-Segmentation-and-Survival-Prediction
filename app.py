from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO
import pickle
from logger import logging

matplotlib.use('Agg')

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model("brats_3d.hdf5",compile=False)

logging.info("model loaded")


with open('Survival_Prediction/survival_pred_model.pkl', 'rb') as f:
        survival_model = pickle.load(f)

def predict(input_data):
    logging.info("Input Image Taken")
    input_data_input = np.expand_dims(input_data, axis=0)
    predictions = model.predict(input_data_input)
    logging.info("Input image prediction done")
    prediction_argmax = np.argmax(predictions, axis=4)[0,:,:,:]
    
    return prediction_argmax


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_view():

    input_file=request.files['input-file']
    input_data = np.load(input_file)

    #total brain pixels

    total_pixels = np.prod(input_data.shape[:-1])  

    #input display

    input_image = input_data[:, :, 55]
    fig, ax = plt.subplots()
    ax.imshow(input_image)
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    input_base64 = base64.b64encode(buffer.getvalue()).decode()

    segmentation = predict(input_data)
 
    age = request.form['age']
    age = float(age)
    
    survival_input=[]   
    survival_input.append(age)

    brain_pixels = np.zeros(4)
    for i in range(4):
        brain_pixels[i] = np.sum(segmentation == i)

    brain_pixels_list = list(brain_pixels)
    print("brain",brain_pixels_list)
    print(segmentation.shape)
    print(type(segmentation))


    for i in brain_pixels_list:
        survival_input.append(i/total_pixels)

    print([survival_input])  

    surival_days=survival_model.predict([survival_input])

    print(surival_days)
    logging.info(f"Total survival days : {surival_days} ")
    fig, ax = plt.subplots()
    ax.imshow(segmentation[:,:, 55])
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    segmentation_base64 = base64.b64encode(buffer.getvalue()).decode()

    return render_template('index.html', input=input_base64, output=segmentation_base64, survival_days=surival_days[0])


if __name__ == '__main__':
    app.run(host="0.0.0.0")
