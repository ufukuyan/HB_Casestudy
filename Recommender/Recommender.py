import numpy as np
import pandas as pd
import json
import sqlite3
import sys
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ProductRecommender:
    """
    A class to represent content-based product recommender.

    Attributes
    ----------
    predictWithPreTrainedModel : bool
        flag to indicate making prediction with pretrained and saved model
    meta_df : pandas dataframe
        pandas dataframe to hold meta data
    cosine_similarity_tfidf : numpy array
        numpy array to hold cosine similarity matrix of tfidf vectorizer
    cosine_similarity_count : numpy array
        numpy array to hold cosine similarity matrix of count vectorizer
    indices : pandas series
        pandas series to hol product id indices

    Methods
    -------
    read_metadata(filename) :
        Read meta data of products
    write_metadata2_sql(filename) :
        Write meta data of product to SQL for future use
    read_turkish_stopwords(filename) :
        read Turkish stopwords for tfidf and count vectorizer
    train_recommender(filename_metadata, filename_turkish_stopwors) :
        train content-based recommender model
    predict_products(product_ids) :
        predict ten similar products of given products product
    save_training_results() :
        save training results for future use
    load_training_results() :
        load previously saved training data
    """ 

    def __init__(self, predictWithPreTrainedModel=True):
        """
        Construct recommender class. If training phase want to be bypassed the saved training data is loaded.

        Parameters
        ----------
            predictWithPreTrainedModel(defaultValue=True) : bool
                flag to indicate making prediction with pretrained and saved model
        """

        self.predictWithPreTrainedModel = predictWithPreTrainedModel
        if self.predictWithPreTrainedModel:
            self._load_training_results()

    def _read_metadata(self, filename):
        """
        Read product meta data. Json and db extensions are supported.
        
        Brand, category, and subcategory and merged as combined_fatures.
        String in  name, brand, category, and subcategory are lowered.

        Parameters
        ----------
            filename : str
                filename and full path of product meta data.
        Returns
        -------
            None
        """
        if filename.endswith('db'):
            conn_meta = sqlite3.connect(filename)
            self.meta_df = pd.read_sql('SELECT * FROM {}'.format('meta'), conn_meta)
        elif filename.endswith('json'):
            try:
                with open(filename, 'r', encoding="utf8") as meta_file:
                    meta_data = json.load(meta_file)
            except OSError as err:
                print("OS error: {0}".format(err))
            except:
                print("Unexpected error:", sys.exc_info()[0])
            else:
                self.meta_df = pd.json_normalize(meta_data, 'meta')
                self.meta_df.drop_duplicates(inplace=True)
                self.meta_df.dropna(subset=['productid'], inplace=True, axis=0)

                self.meta_df['brand'] = self.meta_df['brand'].apply(lambda x: x.lower() if not x is None else None)
                self.meta_df['category'] = self.meta_df['category'].apply(
                    lambda x: x.lower() if not x is None else None)
                self.meta_df['subcategory'] = self.meta_df['subcategory'].apply(
                    lambda x: x.lower() if not x is None else None)
                self.meta_df['name'] = self.meta_df['name'].apply(lambda x: x.lower() if not x is None else None)
                self.meta_df['brand'] = self.meta_df['brand'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
                self.meta_df['category'] = self.meta_df['category'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
                self.meta_df['subcategory'] = self.meta_df['subcategory'].apply(
                    lambda x: re.sub(r'[^\w\s]', '', str(x)))
                self.meta_df['combined_features'] = self.meta_df['brand'] + ' ' + self.meta_df['category'] + ' ' + \
                                                    self.meta_df['subcategory']
                self.meta_df.drop(['brand', 'category', 'subcategory'], axis=1, inplace=True)
        else:
            raise ValueError('Invalid metadata file extension')

    def write_metadata2_sql(self, filename):
        """
        Write product meta data to database. 

        Parameters
        ----------
            filename : str
                filename of saved database.
        Returns
        -------
            None
        """
        filename = filename + '.db'
        try:
            conn_meta = sqlite3.connect(filename)
            self.meta_df.to_sql(name='meta', con=conn_meta, index=False, if_exists='replace')
        except:
            print("Unexpected error:", sys.exc_info()[0])
        finally:
            conn_meta.close()

    def _read_turkish_stopwords(self, filename):
        """
        Read stopwords in Turkish. 

        Parameters
        ----------
            filename : str
                filename of Turkish stopwords.
        Returns
        -------
            None
        """
        if filename.endswith('txt'):
            with open(filename, 'r', encoding="cp1254") as f:
                self.turkish_stop_words = f.read().split()
        else:
            raise ValueError('Invalid metadata file extension')


    def train_recommender(self, filename_metadata, filename_turkish_stopwors):
        """
        Train product recommender with tfidf and count vectorizer.

        Name info is used in tfidf vectorizer.
        combined_features are used in count vectorizer.

        Cosine Similarity  matrices are produced for both vectorizer.
        Training results are dumped to the pickle file for future use.


        Parameters
        ----------
            filename_metadata : str
                filename of product meta data.
            filename_turkish_stopwors :  str
                filename of Turkish stopwords.
        Returns
        -------
            None
        """
        self._read_metadata(filename_metadata)
        self._read_turkish_stopwords(filename_turkish_stopwors)

        vectorizer = TfidfVectorizer(stop_words=self.turkish_stop_words)
        matrix = vectorizer.fit_transform(self.meta_df['name'])
        self.cosine_similarity_tfidf = np.array(cosine_similarity(matrix, matrix))
        self.meta_df = self.meta_df.reset_index()
        self.indices = pd.Series(self.meta_df.index, index=self.meta_df['productid'])

        count_vectorizer = CountVectorizer(stop_words=self.turkish_stop_words)
        count_matrix = count_vectorizer.fit_transform(self.meta_df['combined_features'])
        self.cosine_similarity_count = np.array(cosine_similarity(count_matrix, count_matrix))
        self._save_training_results()

    def _save_training_results(self):
        """
        Save training results as a pickle file.

        Parameters
        ----------
            None
        Returns
        -------
            None
        """
        with open('training_results.pkl', 'wb') as f:
            pickle.dump([self.indices, self.cosine_similarity_tfidf, self.cosine_similarity_count, self.meta_df], f)

    def _load_training_results(self):
        """
        Load training results as a pickle file.

        Parameters
        ----------
            None
        Returns
        -------
            None
        """
        with open('training_results.pkl', 'rb') as f:
            self.indices, self.cosine_similarity_tfidf, self.cosine_similarity_count, self.meta_df = pickle.load(f)

    def predict_products(self, product_ids):
        """
        Predict similar products for give product list.

        Prediction is performed with weighted similarity matrix. 60% percent of count similarity matrix is combined with 40% of tfidf similarity matrix.
        The top ten recommendation with similarity scores are provided.


        Parameters
        ----------
            product_ids : list
                list of products in cart.
        Returns
        -------
            None
        """
        total_scores = []
        product_idxs = []
        for product_id in product_ids:
            idx = self.indices[product_id]
            product_idxs.append(idx)
            weighted_similarity = 0.6 * self.cosine_similarity_count + 0.4 * self.cosine_similarity_tfidf
            sim_scores = list(enumerate(weighted_similarity[idx]))
            total_scores.extend(sim_scores)

        sorted_scores = sorted(total_scores, key=lambda x: x[1], reverse=True)
        product_indices = []
        product_scores = []
        for score in sorted_scores:
            if len(product_indices) == 10:
                break
            if score[0] not in product_idxs and score[0] not in product_indices:
                product_indices.append(score[0])
                product_scores.append(score[1])

        return list(zip(self.meta_df['productid'].iloc[product_indices], product_scores))


if __name__ == '__main__':
    recommender = ProductRecommender(predictWithPreTrainedModel=True)

