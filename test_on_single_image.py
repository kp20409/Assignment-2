# # Testing on Single Image

from tensorflow.keras.models import load_model

# load model
model = load_model(r'models\model2.h5')

model.summary()

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import PIL

image_for_testing = r'Dataset\Training\Fire\resized_frame0.jpg'

test_image = image.load_img(image_for_testing, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = test_image / 255
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

image_show = PIL.Image.open(image_for_testing)
plt.imshow(image_show)
print(result)
if np.argmax(result) == 0:
    plt.title("Fire")
else:
    plt.title("No Fire")

plt.show()
