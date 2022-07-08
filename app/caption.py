class CaptionPreprocessing:
    
    feature_captions = {}
    word_count = {}
    
    def fit(self, path = {}, image_features = {}):
        """
        make dictionary data, image's name as a key and captions as values
        
        example:
            image_1.jpg : ["caption 1", "caption 2", "caption 2"]
        args:
            image_features dictionary -> feature of the image
            path string -> caption path
            
        returns:
            None
        """
        
        # open caption and read line by line
        captions = open(path, 'rb').read().decode("utf-8").split("\n")
        
        # make dictionary
        captions_dict = self.__make_dictionary(data = captions, image_feature = image_features)
        
        # make captions boundary
        captions_boundary = self.__create_boundary(captions_dict)
        
        # count all each word that appears
        word_count = self.__count_word(captions_boundary)
        
        # create index word
        word_indexing = self.__indexing_word(word_count)
        
        # convert caption string to number
        cap_str_to_num = self.__caption_string_to_num(captions_dict, word_indexing)
        
        self.feature_captions = cap_str_to_num
    
    def __make_dictionary(self, data = [], delimiter = ",", image_feature = {}):
        """
        args:
            image_features dictionary -> feature of the image
            delimiter chr -> chr that separate the value, example in csv is , (comma)
            data list -> list of image name and captions splited by delimiter
        retuns:
            dictionary image name as key and captions as value
            
        return example:
            image_1.jpg : ["caption 1", "caption 2", "caption 2"]
        """
        result = {}
        for d in data[1:]:
            
            try:
                
                image_name = d.split(delimiter)[0]
                image_caption = d.split(delimiter)[1]
                
                if image_name in image_feature:
                    if image_name not in result:
                        result[image_name] = [image_caption]
                    else:
                        result[image_name].append(image_caption)
            except:
                pass
        
        return result
    
    def __create_boundary(self, data = {}):
        """
        add boundary where the text is started and when is the text should stoped
        example:
            startofseq i love you endofseq
        
        args:
            data dictionary -> image name as key and captions as values
            
        returns:
            data args but added with boundary
        """
        results = data
        
        for k, v in data.items():
            for tv in v:
                results[k][v.index(tv)] = 'startofseq '+ tv.lower() + ' endofseq'.replace(".", "")
        return results
    
    def __count_word(self, data = {}):
        """
        count each word
        example:
            what : 64 // what appears 64 times
            is : 22 // is appears 22 times
            that : 11 // that appears 11 times
            
        args:
            data dictionary -> key(image name) and value (captions)
        
        returns:
            dictionary
        """
        
        results = {}
        
        for k, v in data.items():
            for tw in v:
                for word in tw.split():
                    if word not in results:
                        results[word] = 0
                    else:
                        results[word] += 1
                        
        return results
    
    def __indexing_word(self, word = {}, treshold = 0):
        """
        give an index to each word if words appearsh more than threshold
        args:
            word dictionary -> key and value data
        returns:
            dicionary
        """
        results = {}
        
        count = 1
        for k, v in word.items():
            if word[k] > treshold:
                results[k] = count
                count += 1

        results["<OUT>"] = len(results)
        self.word_count = results
        
        return results
    
    def __caption_string_to_num(self, caption_string = {}, word_count = {}):
        """
        change caption from string to integer
        args:
            caption_string dictionary -> key and value with original string hat has add boundary
            word_count dictionary -> dictionary that store how much word appears
        returns:
            converted captions to number
        """
        
        results = caption_string
        
        for k, v in caption_string.items():
            for tv in v:
                encoded = []
                for word in tv.split():
                    if word not in word_count:
                        encoded.append(word_count["<OUT>"])
                    else:
                        encoded.append(word_count[word])
                        
                results[k][v.index(tv)] = encoded
        return results
    
    def max_len(self, data = {}):
        """
        get max length sequences of data
        args:
            data dictionary -> key and value pairs
        returns:
            int
        """
        length = 0
        
        for k, v in data.items():
            for tv in v:
                if len(tv) > length:
                    length = len(tv)
        return length
    
    def inverse_feature(self):
        """
        Inverse dictionary from key to value and from value to key
        args:
            None
        returns:
            dictionary
        """
        results = {v : k for k, v in self.word_count.items()}
        return results