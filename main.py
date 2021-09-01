from flask import Flask
from flask import request, current_app, abort
from functools import wraps
from Recommender import ProductRecommender

app = Flask(__name__)
app.config.from_object('settings')


def token_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-TOKEN', None) != current_app.config['API_TOKEN']:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

# Calling prediction method with list of product id.
@app.route('/predict', methods=['POST'])
@token_auth
def predict():
    product_list = request.get_json()
    if not product_list:
        return []
    recommendations = recommender.predict_products(product_list['cart'])
    return {'recommended_products': recommendations}

# Calling training method with full path of product meta data file.
@app.route('/train')
@token_auth
def train():
    filename_turkish_stopwors = 'turkish_stopwords.txt'
    filenames = request.get_json()
    recommender.train_recommender(filenames['filename_metadata'], filename_turkish_stopwors)
    return {"message": "Successfully Trained!", "success": 1}


if __name__ == '__main__':
    recommender = ProductRecommender(predictWithPreTrainedModel=False)
    app.run(debug=True)
