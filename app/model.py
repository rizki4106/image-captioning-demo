import json
import os
from pathlib import Path
import shutil
from zipfile import ZipFile
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, RepeatVector, Embedding, LSTM, TimeDistributed, Activation, Concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize
from tensorflow.keras.applications import ResNet50
from tensorflow import argmax
from .image import ImagePreprocessing
from .caption import CaptionPreprocessing
import numpy as np

class AIModel:

    model = None
    backbone_model = None
    training_history = {}
    inverse_word = {}
    index_word = {}
    maxlen_sequence = 10

    def load_model(self,
                    model_path = "",
                    word_dictionary_path = "",
                    word_index_path = "",
                    conf_path = ""):
        """
        load pretrained model
        args:
            path string -> path to your pre-trained model
        """

        self.model = load_model(model_path)

        # image feature extraction
        backbone = ResNet50(include_top=True)
        output_backbone = backbone.layers[-2].output
        self.backbone_model = Model(inputs=backbone.inputs, outputs=output_backbone)

        # read inverse word
        with open(word_dictionary_path, "r") as word:
            self.inverse_word = json.loads(word.read())
            word.close()

        # read word count and index
        with open(word_index_path, "r") as word:
            self.index_word = json.loads(word.read())
            word.close()

        # read configuration
        with open(conf_path, "r") as word:
            self.maxlen_sequence = int(word.read().split(":")[1])
            word.close()

    def predict(self, image_path = "", output_length = 25):
        """
        predict unseen data from image file
        args:
            image_path string -> image location
        """

        if self.model == None:
            raise Exception("Model can't be null, please load your pre - trained model or train your data")
        else:
            # read image from path
            imgs = img_to_array(load_img(image_path))
            imgs = resize(imgs, (224, 224)).numpy()
            imgs = imgs.reshape(1, 224, 224, 3)
            return self.__make_prediction(imgs, output_length)

    def predict_from_array(self, image_arr = [], output_length = 25):
        """
        predict caption from array of pixel image
        """

        if self.model == None:
            raise Exception("Model can't be null, please load your pre - trained model or train your data")
        else:
            return self.__make_prediction(image_arr, output_length)
    
    def __make_prediction(self, image_arr = [], output_length=25):
        """
        tell the model tu run prediction based on array of pixel image
        and then return the sequence of the text
        args:
            image_arr string -> array pixel values of image
        """
        imgs = self.backbone_model.predict(image_arr).reshape(1, 2048)

        # read inversted dictionary
        
        text_input = ["startofseq"]
        
        count = 0
        captions = ''
        
        while count < output_length:
            
            encoded = []
            
            for i in text_input:
                encoded.append(int(self.index_word[i]))
            
            encoded = [encoded]
            encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=self.maxlen_sequence)
            
            prediction = argmax(self.model.predict([imgs, encoded], verbose=0), axis=1).numpy()[0]
            
            sample_word = self.inverse_word[str(prediction)]
            
            captions = captions + ' ' + sample_word
            
            if sample_word == "endofseq":
                break
                
            text_input.append(sample_word)
            
            count += 1
        
        return captions.replace("endofseq", '').replace("<OUT>", '')


    def train(self, image_path = "", caption_path = "", epochs = 5, batch_size=32, percentage_data = 1.0, steps_per_epoch = None):
        """
        Preprocess the data and then train to the model
        args:
            image_path string -> folder path that store all of the image
            caption_path string -> path to the caption location
            epochs int -> how many times you want to train neural network
            batch_size int
            percentage_data float -> how many percent data that you want to use
        """
        # preprocessing image
        prepimg = ImagePreprocessing()
        prepimg.fit(folder_path=image_path, percentage=percentage_data)
        
        # preprocessing caption
        prepcap = CaptionPreprocessing()
        prepcap.fit(caption_path, prepimg.feature_vectors)


        # caption parameter
        vocab_size = len(prepcap.word_count)
        maxlen = prepcap.max_len(prepcap.feature_captions)

        # write inverse dictionary
        save_model_path = "conf/"

        if not os.path.exists(save_model_path):
            Path(save_model_path).mkdir(exist_ok=True, parents=True)

        with open(os.path.join(save_model_path,  "dictionary.txt"), "w") as files:
            files.write(json.dumps(prepcap.inverse_feature()))
            files.close()

        # save wordcount
        with open(os.path.join(save_model_path,  "word_count.txt"), "w") as files:
            files.write(json.dumps(prepcap.word_count))
            files.close()

        # save configuration
        with open(os.path.join(save_model_path,  "conf.txt"), "w") as files:
            files.write(f'maxlen_sequence:{str(maxlen)}')
            files.close()
        
        #
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        
        # get data transformed
        x_1, x_2, y = self.generate_data(prepimg.feature_vectors, prepcap.feature_captions, maxlen, vocab_size)

        # start learning pattern on data
        self.start_learning(x1=x_1, x2=x_2, y=y, epochs=epochs, batch_size=batch_size, save_model_path = save_model_path, steps_per_epoch=steps_per_epoch)
        
    
    def start_learning(self, x1 = [], x2 = [], y = [], epochs=5, steps_per_epoch=None, batch_size=32, save_model_path = "."):
        """
        Train neural network to learn image and captions
        """
        embedding_size=128
        
        image_model = Sequential([
            Dense(embedding_size, input_shape=(2048, ), activation="relu"),
            RepeatVector(self.maxlen)
        ])
        
        caption_model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=embedding_size, input_length=self.maxlen),
            LSTM(256, return_sequences=True),
            TimeDistributed(Dense(embedding_size))
        ])
        
        inputs = Concatenate()([image_model.output, caption_model.output])
        x = LSTM(128, return_sequences=True)(inputs)
        x = LSTM(512, return_sequences=False)(x)
        x = Dense(self.vocab_size)(x)
        outputs = Activation("softmax")(x)
        
        model = Model(inputs=[image_model.input, caption_model.input], outputs=outputs)
        
        model.compile(
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            optimizer=RMSprop()
        )
        
        history = model.fit([x1, x2], y,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 steps_per_epoch=steps_per_epoch
                                 )
        self.training_history = history
        
        model.save(os.path.join(save_model_path, "trained_model.h5"))
        
    def generate_data(self, image, captions, maxlenseq, vocabsize):
        """
        generate x1 (image feature), x2 (caption feature), y (next word in sentences)
        """
        x = []
        y_in = []
        y_out = []

        for k, v in captions.items():
            for vv in v:
                try:
                    for i in range(1, len(vv)):

                        input_sequence = [vv[:i]]
                        output_sequence = vv[i]

                        input_sequence = pad_sequences(input_sequence, maxlen=maxlenseq, padding='post', truncating='post')[0]
                        output_sequence = to_categorical([output_sequence], num_classes=vocabsize)[0]

                        y_in.append(input_sequence)
                        y_out.append(output_sequence)
                        x.append(image[k])
                except:
                    pass

        return np.array(x), np.array(y_in, dtype="float64"), np.array(y_out, dtype="float64")

    def create_zip(self, filename = "", source = ""):
        """
        create zipfile from folder
        args:
            filename string -> output file name (without .zip)
            source string -> folder name that you want to zip
        """
        shutil.make_archive(filename, "zip", source)

        print(f'{filename}.zip')

    def unzip(self, filename, target):
        """
        unzip file
        args:
            filename string -> zip's file name
            target string -> path where do you want to store extracted file
        """

        with ZipFile(filename) as zip:
            zip.extractall(target)
            zip.close()
        
        print(f'{filename} unzipped to {target}')