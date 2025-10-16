from flask import Flask , request , jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('mnist_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data =request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    image = np.array(data['image'])
    image =image.reshape(1, 784).astype('float32') / 255.0
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return jsonify({
        'predicted_class': int(predicted_class),
        'probabilities': prediction.tolist()
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 5000)
