import utils as util
from sklearn.metrics import classification_report


# PHASE 1, Training on Data of Client to learn there liked and dislikes with multiple round. 
#Initial Round
list_of_models, list_of_test_data, list_of_feature_names, list_of_client_object = util.runRound(util.initialExecute)

list_of_reports = []
for i in range(0, len(list_of_models)):
    prediction = list_of_models[i].predict(list_of_test_data[i][0])
    report = classification_report(prediction,list_of_test_data[i][1], output_dict = True)
    list_of_reports.append(report)

average_of_report = util.report_average(list_of_reports)
print(average_of_report)

#Aggregating Models from Clients
model = util.makeEstimatorForStacking(list_of_models)
list_of_models, list_of_test_data, list_of_feature_names = util.runRound(util.execute, args=(list_of_client_object ,model))

#Second Round
list_of_reports = []
for i in range(0, len(list_of_models)):
    prediction = list_of_models[i].predict(list_of_test_data[i][0])
    report = classification_report(prediction,list_of_test_data[i][1], output_dict = True)
    list_of_reports.append(report)

average_of_report = util.report_average(list_of_reports)
print(average_of_report)

# PHASE 2, Until now we have learned the clients likes and now we are clustering likes of all client and recommending new products to client securely.
list_of_liked_products = util.clusteringDataCollections(list_of_client_object) # Collecting Liked products from clients .
concatenate_data_frame = util.concatenateDataFrame(list_of_liked_products) # Joining Data Collected to collective Data Processing.
product_dict = util.makeReplaceDictofDataFrame(concatenate_data_frame) # Creating Encoding Dictionary to compute data effeciently.
inverted_product_dict = util.invertDictionary(product_dict) # Created Interted Dictionary to Decode Data after Clustering.
list_of_products = util.clusteringDataProcessing(list_of_liked_products, product_dict) # Encoding Data here using encoder.
encoded_list_of_potentail_likes = util.createCluster(list_of_products) # Making Intersecting and Finding Common Likes against all Clients. 
decoded_list_of_potentail_likes = util.decodePotentailLikedProducts(encoded_list_of_potentail_likes, inverted_product_dict) # Decoding the List of Liked Products for sharing it to client.
util.sendPotentialLikedProducttoClient(list_of_client_object, decoded_list_of_potentail_likes) # Sharing the potenital Likes back to respective cleint