from tensorflow.image import resize
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
import os

class ImagePreprocessing:
    
    __model = None
    feature_vectors = {}
    
    def __init__(self):
        
        resnet = ResNet50(include_top=True)
        output_resnet = resnet.layers[-2].output
        
        self.__model  = Model(inputs=resnet.inputs, outputs=output_resnet)
    
    def extract_feature(self, image_path = "", image_size=(224, 224)):
        """
        Extract image feateature using pre - trained model
        in this case using ResNet50
        
        args:
            image_path string -> location of the image
            image_size tupple -> (width, height) of the image
        return:
            image name, list of image features
        """
        image = resize(img_to_array(load_img(image_path)), image_size).numpy()
        
        # (1, 224, 224, 3)
        image = image.reshape((1,) + image_size + (3,))
        
        # extract featurte using pre - trained model
        feature = self.__model.predict([image], verbose=0)[0].tolist()
        
        return image_path.split("/")[-1], feature
    
    def fit(self, folder_path = "", image_size = (224, 224), percentage = 1.0):
        """
        Fit the data into preprocessing handler and let them learn your data
        args:
            folder_path string -> where you store  your image
            image_size tupple -> (width, heigh) of the image that you want
            percentage float -> how many percent do you want to use the data (default 100%)
        returns:
            None
        """
        
        images = os.listdir(folder_path)
        count_files = int(percentage * len(images))

        print(f'{count_files} files {percentage * 100}% of data used for training')
        
        image_features = {}
        
        for i, img in enumerate(images[:count_files]):
            image_name, feature_vector = self.extract_feature(os.path.join(folder_path, img), image_size)
            if img not in image_features:
                image_features[img] = feature_vector
            print(f'extracting image feature {round((i / len(images[:count_files])) * 100, 2)}%')
        
        self.feature_vectors = image_features
        