import sys
import os
from io import BytesIO
from PIL import Image
from PIL import ImageFilter
import numpy as np
from flask import Flask, request, render_template, send_file
from flask_uploads import UploadSet, configure_uploads, IMAGES
from utils.notebook_utils import GANmut

app = Flask(__name__, static_folder='static')

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/image'
configure_uploads(app, photos)

G = GANmut(G_path='./learned_generators/gaus_2d/1800000-G.ckpt', model='gaussian')

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        image_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        if filename.endswith(".png"):
            output_image_name = f'{filename.split(".")[0]}.png'
        else:
            output_image_name = f'{filename.split(".")[0]}.jpg'
            
        emotion = request.form.get("emotions")
        return emotion_edit(image_path,emotion, output_image_name)
    return 'No file uploaded'

def emotion_edit(img_path, emotion, output_image_name):
    x, y = get_emotion_coordinates(emotion)
    edited_img = G.emotion_edit(img_path, x=x, y=y, save=True)
    edited_img = Image.fromarray((edited_img).astype(np.uint8))
    edited_img_path = os.path.join(app.static_folder, 'edited_images', output_image_name)
    edited_img.save(edited_img_path, format='JPEG', optimize=True, quality=100)
    orig_img_url = os.path.join('static','image', output_image_name)
    edited_img_url = os.path.join('static', 'edited_images', output_image_name)
  
    return render_template('emotion_edit_result.html', orig_img_url=orig_img_url, edited_img_url=edited_img_url , value = emotion)

def get_emotion_coordinates(emotion):
    emotion_coordinates = {
        '0': (-0.2812, -0.4329),
        '1': (0.3152, -0.1149),
        '2': (-1.0000, -0.6416),
        '3': (0.9617, -0.4510),
        '4': (-0.1805, -1.0000),
        '5': (-0.9823, -0.9984),
        '6': (0.8572, -0.9997),
    }
    return emotion_coordinates.get(emotion, (0,0)) 

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    