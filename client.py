#FILE IMPORT
import constants as const
import utils as util

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
#SKLEARN IMPORTS

class ClientSentiment:

    def transformToDataframe(self, list_of_products_sentiment):
        transform_structure = {}
        transform_structure["sentence"] = list(list_of_products_sentiment.keys())
        transform_structure["sentiment"] = list(list_of_products_sentiment.values())

        return transform_structure

    def dataFrameProcess(self, data_frame):
        data_frame = data_frame[data_frame['Score'] != 3]
        data_frame['sentiment'] = data_frame['Score'].apply(lambda rating : +1 if rating > 3 else -1)
        data_frame = data_frame.dropna(subset=['Summary'])
        return data_frame

    def splitDataSet(self, data_frame):
        index = data_frame.index
        data_frame['random_number'] = np.random.randn(len(index))
        train = data_frame[data_frame['random_number'] <= 0.8]
        test = data_frame[data_frame['random_number'] > 0.8]
        self.train = train
        self.test = test
        return train, test


    def dataPreprocessing(self):
        filename = const.DATA_SET_NAME
        if filename == "data.txt":
            list_of_products_sentiment = util.readDataSet()
            transformed_list_of_products_sentiment = self.transformToDataframe(list_of_products_sentiment)
            data_frame = pd.DataFrame.from_dict(transformed_list_of_products_sentiment)
        else:
            data_frame = util.readDataSet()
            data_frame = util.partitionDataFrame(data_frame)
            data_frame = self.dataFrameProcess(data_frame)
            data_frame['sentence'] = data_frame['Summary'].apply(util.removePunctuation)
            train, test = self.splitDataSet(data_frame)
        
        (x_train, y_train), (x_test, y_test), vectorizing_feature_names = util.vectorizing_data_frame(train, test)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.vectorizing_feature_names = vectorizing_feature_names
        return x_train, y_train


    def recieveInitialModel(self):
        self.intial_logistic_regression_model = util.get_intial_model_from_server()

    def recieveModel(self, trained_logistic_regression_model):
        self.trained_logistic_regression_model = trained_logistic_regression_model

    def trainModel(self):
        x_train = self.x_train
        y_train = self.y_train
        trained_logistic_regression_model = self.intial_logistic_regression_model.fit(x_train, y_train)
        self.trained_logistic_regression_model = trained_logistic_regression_model
        self.model = trained_logistic_regression_model

    def sendModelAndTestData(self):
        return self.model, (self.x_test, self.y_test), self.vectorizing_feature_names

    def makeLikeProductList(self):
        model = self.trained_logistic_regression_model
        test = self.test
        predictions = model.predict(self.x_test)
        test["predictions"] = predictions
        test = test[test['predictions'] == 1]
        self.liked_products = test["ProductId"]
        return self.liked_products

    def recievePotenitalLikedProducts(self, list_of_potential_likes):
        print("Received {} numbers of potentail liked Products for {}".format(len(list_of_potential_likes), self))