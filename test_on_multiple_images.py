
from tensorflow.keras.models import load_model

# load model
model = load_model(r'models\model2.h5')

model.summary()

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import PIL

# # Testing on Multiple images

from IPython.display import display


list_result=[]
def process_images():
    for i in os.listdir(r'mixed'):
        images=r'mixed/'+i
        print(images)
        #display(images)

        test_image=image.load_img(images,target_size=(254,254))
        test_image=image.img_to_array(test_image)
        test_image=test_image/255
        test_image=np.expand_dims(test_image,axis=0)
        result=model.predict(test_image)
        list_result.append(result)
        image_show=PIL.Image.open(images)
        plt.imshow(image_show)
        if np.argmax(result)==0:
            plt.title("Fire")
        else:
            plt.title("No Fire")

        plt.show()

process_images()
