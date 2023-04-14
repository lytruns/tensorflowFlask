import os
import json
import tensorflow as tf
import base64
import numpy as np
from PIL import Image
import io
from flask import Flask, request, jsonify

img_height = 180
img_width = 180
model = tf.keras.models.load_model('modelo74.h5')  # Load the model from the saved file
class_names = ['bipolaris_maydis', 'cercospora_kikuchii', 'colletotrichum_dematium', 'colletotrichum_graminicola', 'corynespora_cassicola', 'diplodia_macrospora', 'downy_mildew', 'erysiphe_diffusa', 'exserohilum_turcicum', 'fitotoxicidade_do_cobre', 'maize_bushy', 'mildew_rust', 'myrothecium_roridum', 'phaeosphaeria_maydis', 'phakopsora_pachyrhizi', 'phialophora_gregata', 'physarum_sp', 'physoderma_brown', 'physopella_zeae', 'phytophthora_sojae', 'pseudomonas_savastanoi', 'puccinia_polysora', 'septoria_glycines']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Lê a imagem enviada na requisição
    img_data = request.get_data()
    image = Image.open(io.BytesIO(img_data))
    # Resize the image to the desired dimensions
    image = image.resize((img_width, img_height))
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)
    # Convert the NumPy array to a batch of size 1
    image_array = np.expand_dims(image_array, axis=0)
    #image_array = image_array.astype('float32') / 255.0
    
    # Predict with the model
    predictions = model.predict(image_array)[0]
    top_predictions = predictions.argsort()[-5:][::-1]
    response = []
    
    # Retorna as previsões
    for i in top_predictions:
        label = class_names[i]
        confidence = float(predictions[i])
        response.append({'label': label, 'confidence': confidence})
        
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

