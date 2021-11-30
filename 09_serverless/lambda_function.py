from io import BytesIO
from urllib import request

from PIL import Image

import numpy as np

import tflite_runtime.interpreter as tflite
#import tensorflow.lite as tflite

# model for local testing: model.tflite
# model for docker: cats-dogs-v2.tflite
interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(url):
    download = download_image(url)
    img = prepare_image(download, (150, 150))
    x = np.array(img, dtype='float32')
    x = x/ 255
    X = np.array([x])
    return X

def predict(url):
    X = preprocess(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    actual_pred = preds[0].tolist()

    return actual_pred

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result