# Secure-Privacry-Preserving-Framework-for-Recommendation-System

This is a Implemetation for a Research Paper. 

You can run experiment by executing in terminal 
```
python3 script.py
```

There are mainly 2 phase to this script. They are as follow:

# Phase 1
> Server is responsible to share general logistic regression model to user.

> Client Device Train model on the general model using user data and share back to server.

> Server Aggregates the models and setup a round 2 of training.

> Where client participate and train the new model and use it on their system. 

# Phase 2
> Client device use new model to learn like and dislikes of user. 

> Then liked products are encrypted with a key( remaining to implement ) and share it to server. 

> Server recieve encrypted liked products and perform clustering over the encrypted data to get more products clients might like.

> Server sends back more products to client device that user may like.
