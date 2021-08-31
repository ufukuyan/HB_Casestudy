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
    def __init__(self, predictWithPreTrainedModel=True):
        self.predictWithPreTrainedModel = predictWithPreTrainedModel
        if self.predictWithPreTrainedModel:
            self.load_training_results()

    def read_metadata(self, filename):
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
        filename = filename + '.db'
        try:
            conn_meta = sqlite3.connect(filename)
            self.meta_df.to_sql(name='meta', con=conn_meta, index=False, if_exists='replace')
        except:
            print("Unexpected error:", sys.exc_info()[0])
        finally:
            conn_meta.close()

    def read_turkish_stopwords(self, filename):
        if filename.endswith('txt'):
            with open(filename, 'r', encoding="cp1254") as f:
                self.turkish_stop_words = f.read().split()
        else:
            raise ValueError('Invalid metadata file extension')


    def train_recommender(self, filename_metadata, filename_turkish_stopwors):
        self.read_metadata(filename_metadata)
        self.read_turkish_stopwords(filename_turkish_stopwors)

        vectorizer = TfidfVectorizer(stop_words=self.turkish_stop_words)
        matrix = vectorizer.fit_transform(self.meta_df['name'])
        self.cosine_similarity_tfidf = np.array(cosine_similarity(matrix, matrix))
        self.meta_df = self.meta_df.reset_index()
        self.indices = pd.Series(self.meta_df.index, index=self.meta_df['productid'])

        count_vectorizer = CountVectorizer(stop_words=self.turkish_stop_words)
        count_matrix = count_vectorizer.fit_transform(self.meta_df['combined_features'])
        self.cosine_similarity_count = np.array(cosine_similarity(count_matrix, count_matrix))
        self.save_training_results()

    def save_training_results(self):
        with open('training_results.pkl', 'wb') as f:
            pickle.dump([self.indices, self.cosine_similarity_tfidf, self.cosine_similarity_count, self.meta_df], f)

    def load_training_results(self):
        with open('training_results.pkl', 'rb') as f:
            self.indices, self.cosine_similarity_tfidf, self.cosine_similarity_count, self.meta_df = pickle.load(f)

    def predict_products(self, product_ids):
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
    recommender = ProductRecommender()

