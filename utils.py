#FILE IMPORTS
import server as sr
import client as cl
import constants as const
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from threading import Thread
from sklearn.ensemble import VotingClassifier


def readDataSet():
    filename = const.DATA_SET_NAME
    if filename == "data.txt":
        list_of_products_sentiment = {}
        file_instance = open(filename, 'r')
        dataset = file_instance.read()
        dataset = dataset.split(",")
        for i in range(0, len(dataset)-1, 2):
            list_of_products_sentiment[dataset[i]] = int(dataset[i+1])

        return list_of_products_sentiment

    else:
        data_frame = pd.read_csv(filename)

        return data_frame

def vectorizing_data_frame(train, test):
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = vectorizer.fit_transform(train['Summary'])
    test_matrix = vectorizer.transform(test['Summary'])
    X_train = train_matrix
    X_test = test_matrix
    y_train = train['sentiment']
    y_test = test['sentiment']
    return (X_train, y_train), (X_test, y_test), vectorizer.get_feature_names_out()

def get_intial_model_from_server():
    intial_logistic_regression_model = sr.makeModel()
    return intial_logistic_regression_model

def get_model_from_client(client_object):
    model, (x_test, y_test), vectorizing_feature_names =  client_object.sendModelAndTestData()
    return model, (x_test, y_test), vectorizing_feature_names

def removePunctuation(text):
    cleanText = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return cleanText

def partitionDataFrame(data_frame):
    part_data_frame = data_frame.sample(frac = const.DATA_SET_PARTITION_SIZE)
    rest_data_frame = data_frame.drop(part_data_frame.index)
    return rest_data_frame

def report_average(reports):
    mean_dict = dict()
    for label in reports[0].keys():
        dictionary = dict()

        if label in 'accuracy':
            mean_dict[label] = round(sum(d[label] for d in reports) / len(reports), 2) 
            continue

        for key in reports[0][label].keys():
            dictionary[key] =  round(sum(d[label][key] for d in reports) / len(reports),2)
        mean_dict[label] = dictionary

    return pd.DataFrame(mean_dict)

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def runRound(function_name, args=[]):
    list_of_threads = []
    if len(args) > 0:
        list_of_client_object = args[0]
        model = args[1]
        for i in range(0, const.NUMBER_OF_CLIENTS):
            twrv = ThreadWithReturnValue(target=function_name, args = (list_of_client_object[i], model))
            twrv.start()
            list_of_threads.append(twrv)
        list_of_models = []
        list_of_test_data = []
        list_of_feature_names = []
        for thread in list_of_threads:
            model, (x_test, y_test), vectorizing_feature_names  = thread.join()
            list_of_models.append(model)
            list_of_test_data.append((x_test, y_test))
            list_of_feature_names.append(vectorizing_feature_names)
        return list_of_models, list_of_test_data, list_of_feature_names

    else:
        for i in range(0, const.NUMBER_OF_CLIENTS):
            twrv = ThreadWithReturnValue(target=function_name)
            twrv.start()
            list_of_threads.append(twrv)
        list_of_models = []
        list_of_test_data = []
        list_of_feature_names = []
        list_of_client_object = []
        for thread in list_of_threads:
            model, (x_test, y_test), vectorizing_feature_names, client  = thread.join()
            list_of_models.append(model)
            list_of_test_data.append((x_test, y_test))
            list_of_feature_names.append(vectorizing_feature_names)
            list_of_client_object.append(client)
        return list_of_models, list_of_test_data, list_of_feature_names, list_of_client_object

def initialExecute():
    #CLIENT PART
    client = cl.ClientSentiment()
    client.dataPreprocessing()
    client.recieveInitialModel()
    client.trainModel()
    #SERVER PART
    model, (x_test, y_test), vectorizing_feature_names  = sr.recieveModelFromClient(client)
    return model, (x_test, y_test), vectorizing_feature_names, client

def makeEstimatorForStacking(list_of_models):
    estimator = []
    i = 0
    for model in list_of_models:
        estimator.append(("model{}".format(i), model))
        i += 1
    voting_clf = VotingClassifier(estimator, voting='soft')
    return voting_clf

def execute(client, model):
    #CLIENT PART
    client.dataPreprocessing()
    client.recieveModel(model)
    client.trainModel()
    #SERVER PART
    model, (x_test, y_test), vectorizing_feature_names  = sr.recieveModelFromClient(client)
    return model, (x_test, y_test), vectorizing_feature_names

def clusteringDataCollection(client):
    return sr.receiveLikeProductList(client)

def concatenateDataFrame(list_of_data_frame):
    return pd.concat(list_of_data_frame)

def makeReplaceDictofDataFrame(data_frame):
    return dict(zip(data_frame, range(len(data_frame))))

def invertDictionary(product_dict):
    inv_product_dict = {v: k for k, v in product_dict.items()}
    return inv_product_dict

def common(list_of_client_1, list_of_client_2):
    set_of_client_1 = set(list_of_client_1)
    set_of_client_2 = set(list_of_client_2)
    return set_of_client_1 & set_of_client_2 if set_of_client_1 & set_of_client_2 else []

def unCommon(list1, list2):
    result = [i for i in list1 if i not in list2]
    return set(result)

def calculateWeight(length_of_client_like_products, len_of_common_products):
    return round(float( 1 - ((length_of_client_like_products - len_of_common_products)/length_of_client_like_products)), 2)

def findMaxWeight(input_list):
    max_value = max(input_list)
    index_value = [index for index in range(len(input_list)) if input_list[index] == max_value]
    return index_value

def partition(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i+size]

def createClustersHelper(part_list_of_products, list_of_products):
    new_list = []
    for i in range(0, len(part_list_of_products)):
        for j in range(0, len(list_of_products)):
            if part_list_of_products[i] == list_of_products[j]:
                continue
            client_list = []
            dict = {}
            dict["potential_liked_products"] = common(part_list_of_products[i], list_of_products[j])
            dict["liked_products"] = unCommon(part_list_of_products[i], dict["potential_liked_products"])
            client_list.append(dict)
            j += 1
        i += 1
        new_list.append(client_list)
    return new_list

def createCluster(list_of_products):
    thread_list_of_products = list(partition(list_of_products, 100))
    list_of_thread_object = []
    for product_list in thread_list_of_products:
        twrv = ThreadWithReturnValue(target=createClustersHelper, args = (product_list, list_of_products))
        twrv.start()
        list_of_thread_object.append(twrv)

    new_list = []
    for thread in list_of_thread_object:
        thread_new_list = thread.join()
        new_list.extend(thread_new_list)
    
    return new_list

def clusteringDataCollections(list_of_client_object):
    list_of_liked_products = []
    for client in list_of_client_object:
        list_of_liked_products.append(clusteringDataCollection(client))

    return list_of_liked_products

def clusteringDataProcessing(list_of_liked_products, product_dict):
    list_of_products = []
    for product in list_of_liked_products:
        product = product.replace(product_dict)
        list_of_products.append(product.values.tolist())

    return list_of_products

def decodePotentailLikedProducts(encoded_list_of_potentail_likes, inverted_product_dict):
    list_of_potential_likes = []
    for data in encoded_list_of_potentail_likes:
        client_list_of_potential_like = []
        for potential_like_product in data[0].get("potential_liked_products"):
            client_list_of_potential_like.append(inverted_product_dict.get(potential_like_product))
        list_of_potential_likes.append(client_list_of_potential_like)
    return list_of_potential_likes

def sendPotentialLikedProducttoClient(list_of_client_object, list_of_products):
    for i in range(0 , len(list_of_client_object)):
        sr.sendPotenitalLikedProducts(list_of_client_object[i], list_of_products[i])
