import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import os
import base64
from PIL import Image

app = Flask(__name__)

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def dark_channel_prior(img):
    length, width, colors = np.shape(img)
    dark_channel = np.zeros(shape=(length, width))
    for y in range(0, length):
        for x in range(0, width):
            dark_channel[y][x] = min(img[y][x][:])
    return dark_channel


def calculate_atmospheric_light(img):
    gray_img = rgb2gray(img)
    dark_channel = dark_channel_prior(img)
    pixels_to_select = int(len(dark_channel) * len(dark_channel[0]) \
                           * 0.2)
    dark_channel_pixels = []
    for y in range(0, len(dark_channel)):
        for x in range(0, len(dark_channel[0])):
            dark_channel_pixels.append((dark_channel[y][x], y, x))
    dark_channel_pixels = sorted(dark_channel_pixels, key=lambda x: x[0], reverse=True)
    best_pixels = dark_channel_pixels[0:pixels_to_select]
    A = -1
    atmospheric_array = []
    for pixel in best_pixels:
        dark_value, y, x = pixel
        if (A < max(A, gray_img[y][x])):
            atmospheric_array = img[y][x]
            A = max(A, gray_img[y][x])

    return atmospheric_array


def get_transmission(img, w = 0.9):
    length, width, colors = np.shape(img)
    transmission_matrix = np.zeros(shape=(length, width))
    atmospheric_array = calculate_atmospheric_light(img)
    for y in range(0, length):
        for x in range(0, width):
            array = img[y][x][:]
            min_value = min(array)
            min_index = -1;
            for i in range(0, 3):
                if(array[i] == min_value):
                    min_index = i;
                    break;
            transmission_matrix[y][x] = 1 - w * min_value / float(atmospheric_array[min_index])
    return transmission_matrix


def get_radiance(img, t0=0.1):
    length, width, colors = np.shape(img)
    atmospheric_light = calculate_atmospheric_light(img)

    atmospheric_light = atmospheric_light.astype(np.float)
    transmission = get_transmission(img)
    transmission = transmission.astype(np.float)
    J = np.zeros(shape=(length, width, colors))
    J = J.astype(np.float)

    for y in range(0, length):
        for x in range(0, width):
            J[y][x] = np.subtract(img[y][x], atmospheric_light)
            J[y][x] = np.divide(J[y][x], (max(transmission[y][x], t0)))
            J[y][x] = np.add(J[y][x], atmospheric_light)
    return J


def clip(photo):
    for y in range(0, len(photo)):
        for x in range(0, len(photo[0])):
            for c in range(0, 3):
                if (photo[y][x][c] > 1):
                    photo[y][x][c] = float(1)
                elif(photo[y][x][c] < 0):
                    photo[y][x][c] = 0;
    return photo

#
@app.route('/send', methods=['POST'])
def get_names():
    ret_string = ''
    if request.method == 'POST':
        base = request.form.get('image')
        comma = base.index(',')
        base = base[comma + 1::]
        image = base64.b64decode(base)
        fh = open("imageToSave.jpg", "wb")
        fh.write(image)
        fh.close()
        im = Image.open("imageToSave.jpg")
        rgb_im = np.array(im)
        plt.imsave("imageToSave.jpg", rgb_im)
        rgb_im = plt.imread("imageToSave.jpg")
        return_image = process(rgb_im)
        plt.imsave("imageToSave.jpg", return_image)
        with open("imageToSave.jpg", "rb") as f:
            ret_string = base64.b64encode(f.read())
            ret_string = "data:image/jpeg;base64," + ret_string.decode("utf-8")
        f.close()
    return ret_string, 200


def process(photo):
    J = get_radiance(photo)/ 255;
    J = clip(J);
    return J


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max_age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

@app.route('/')
def render_webpage():
     return render_template("HeWebApp.html")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)