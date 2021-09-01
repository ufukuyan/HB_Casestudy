# HB_Casestudy

## Description

This is a production-ready, but very simple, content-based recommendation engine that computes similar items based on text product name, brand, category, and subcategory. 

maim.py contains the two endpoints:

1. /train -- calls train() which computes item similarities using TF-IDF and Count Vectorizors.

2. /predict -- given an product_id list, returns the most similar ten products.

## Try it out!

Create a new virtualenv with the needed dependencies:

> pip3 install requirenments.txt

Activate virtual environment.

Run main.py

To predict with pre-trained and saved model, assign predictWithPreTrainedModel as True in main.py script.

Then, in a separate terminal window, train the engine:

> curl -X GET -H "X-API-TOKEN: FOOBAR1" -H "Content-Type: application/json; charset=utf-8" http://127.0.0.1:5000/train -d '{"filename_metadata": "meta.json"}'

And make a prediction!

> curl -X POST -H "X-API-TOKEN: FOOBAR1" -H "Content-Type: application/json; charset=utf-8" http://127.0.0.1:5000/predict -d '{"cart": ["HBV00000PV7JO","HBV00000AX6LR","HBV00000NE0UQ"] }'

This recommendation system return list of top 10 similar products with similarity scores.


## Content-based Recommendation Engine
In a very general way, recommender systems are algorithms aimed at suggesting relevant items to users by using their past experiences and preferences. 

Collaborative Filtering matches persons with similar interests and provides recommendations based on this matching. However, user information was not provided in the data. 

According to my literature research, content-based recommendation and session-based recommendation system can be utilized in this case study.

Content-Based Filtering suggests similar items based on a particular item. This system uses item metadata, such as name, brand, description, category, etc. to make these recommendations. 

Session-based recommendation systems suggest top-K items based on the sequence of items clicked or added to the cart so far.
Session-based kNN, Gru4Rec, IIRNN, SR-GNN algorithms are widely used Session-based recommendation algorithms.

Due to the time constraints, I can only implement a content-based recommender by just using provided metadata. 
I merge brand, category, and subcategory information. I used the merged information in the Count vectorizer and calculate the cosine similarity matrix.
However, I deiced to use TF-IDF Vectorizer with product name information and calculate cosine similarity matrix.
Finally, I combined both cosine similarity matrices with giving 60% weight to the Count vectorizer originated cosine similarity matrix.

Pros:
  - I use TF-IDF vectorizer for calculating product name similarity to penalize common words 
  - I use a Count vectorizer for calculating similarity in merged brand, category, and subcategory information.
  - This recommendation algorithm can be trained without needing extra time and computation resources.
  - This algorithm can predict the top 10 similar products by getting a product list.

Cons:
  - Provided event data cannot be used.
  - Deep learning techniques can be used with session-based recommendation systems.
  
