from flask import Flask, render_template, request, send_file
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Function to colorize a grayscale image
def colorize_image(image):
    proto_txt = './models/colorization_deploy_v2.prototxt'
    caffemodel = './models/colorization_release_v2.caffemodel'

    try:
        net = cv2.dnn.readNetFromCaffe(proto_txt, caffemodel)
        pts = np.load('./models/pts_in_hull.npy')
        print("Network successfully loaded!")
    except cv2.error as e:
        print(f"CV2 Error: {e}")
        return None, None

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return image, colorized

# Main route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and display the result
@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        original, colorized = colorize_image(Image.open(uploaded_file))

        # Save colorized image to a BytesIO object
        colorized_io = BytesIO()
        Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)).save(colorized_io, 'JPEG')
        colorized_io.seek(0)

        return send_file(colorized_io, mimetype='image/jpeg')

    return "Error: No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)
