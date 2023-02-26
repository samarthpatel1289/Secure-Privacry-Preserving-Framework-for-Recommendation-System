#FILE IMPORT
import constants as const
import utils as util


#SKLEARN IMPORTS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def makeModel():
    logistic_regression_model = LogisticRegression(max_iter=1000)
    return logistic_regression_model

def recieveModelFromClient(client_object):
    trained_logistic_regression_model, (x_test, y_test), vectorizing_feature_names  = util.get_model_from_client(client_object)
    return trained_logistic_regression_model, (x_test, y_test), vectorizing_feature_names

def aggregationModel(logistic_regression_model, coef, intercept):
    logistic_regression_model.coef_ = coef
    if logistic_regression_model.fit_intercept:
        logistic_regression_model.intercept_ = intercept
    return logistic_regression_model

def receiveLikeProductList(client_object):
    liked_products = client_object.makeLikeProductList()
    return liked_products

def sendPotenitalLikedProducts(client_object, list_of_potential_liked_products):
    client_object.recievePotenitalLikedProducts(list_of_potential_liked_products)