import pathlib
import numpy as np
import tensorflow as tf

batch_size = 32
img_height = 180
img_width = 180

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image.
img_path = pathlib.Path('image_to_predict_3.jpg')

img = tf.keras.utils.load_img(
    img_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# Set the tensor data for the input tensor.
interpreter.set_tensor(input_details[0]['index'], img_array)

# Perform inference.
interpreter.invoke()

# Get the output tensor data.
output_data = interpreter.get_tensor(output_details[0]['index'])

# Compute softmax activation.
score_lite = tf.nn.softmax(output_data)

# Print the prediction result.
class_names = ['Bed', 'Chair', 'Sofa']
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)
