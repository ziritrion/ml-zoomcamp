from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite

interpreter = tflite.Interpreter(model_path='model.tflite')
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
    img = prepare_image(download, (299, 299))
    x = np.array(img, dtype='float32')
    x /= 127.5
    x -= 1.
    X = np.array([x])
    return X

def predict(url):
    X = preprocess(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    actual_pred = preds[0].tolist()

    return actual_pred

def decode_predictions(prediction):
    labels = [
        'apple',
        'banana',
        'beetroot',
        'bell pepper',
        'cabbage',
        'capsicum',
        'carrot',
        'cauliflower',
        'chilli pepper',
        'corn',
        'cucumber',
        'eggplant',
        'garlic',
        'ginger',
        'grapes',
        'jalepeno',
        'kiwi',
        'lemon',
        'lettuce',
        'mango',
        'onion',
        'orange',
        'paprika',
        'pear',
        'peas',
        'pineapple',
        'pomegranate',
        'potato',
        'raddish',
        'soy beans',
        'spinach',
        'sweetcorn',
        'sweetpotato',
        'tomato',
        'turnip',
        'watermelon'
    ]
    pred_dict = dict(zip(labels, prediction))
    return pred_dict

def short_prediction(pred_dict):
    return max(pred_dict, key=pred_dict.get)

def lambda_handler(event, context):
    url = event['url']
    short = event['short']
    prediction = predict(url)
    result = decode_predictions(prediction)
    if short:
        result = short_prediction(result)
    return result