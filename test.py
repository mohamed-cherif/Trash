import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

def convert_image_dtype(img, dtype=np.float32):
  """Converts an image to a specified data type.

  Args:
    img: A numpy array representing the image.
    dtype: The desired data type.

  Returns:
    A numpy array representing the image in the specified data type.
  """

  img = img.astype(dtype)
  if dtype == np.float32:
    img /= 255.0
  return img

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=r"model.tflite")

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

# input details
print(input_details)
# output details
print(output_details)
file = "test.jpg"
img = np.array(Image.open(file))
img = convert_image_dtype(img, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], [img])
# run the inference
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("For file {}, the output is {}".format(file.stem, output_data))
