import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LSHForest
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import csr_matrix,vstack,hstack
from sklearn.feature_extraction import DictVectorizer
from .util import *
# ## Read raw data as lines

raw_train = pd.DataFrame([line for line in open('../data/classification_train.tsv', encoding='utf8')], columns=['line'])
raw_test = pd.DataFrame([line for line in open('../data/classification_blind_set_corrected.tsv', encoding='utf8')],
                        columns=['line'])

# ## Extract category and brand from raw data
train = raw_train.line.str.extract(r'(.*)\t(\d+)\t(\d+)$', expand=True)
train.columns = ['product_title', 'brand_id', 'category_id']
train = train.dropna()
train.loc[:, ['brand_id', 'category_id']] = train.loc[:, ['brand_id', 'category_id']].astype(int)

test = raw_test.line.str.extract(r'(.*)\t(-?\d+)$', expand=True)
test.columns = ['product_title', 'category_id']
test = test.dropna()
test.loc[:, ['category_id']] = test.loc[:, ['category_id']].astype(int)


def submission_2():
    category_wise_counts = train.groupby(['category_id', 'brand_id']).size().reset_index()
    category_wise_counts.columns = ['category_id', 'brand_id', 'size']
    category_wise_counts = category_wise_counts.sort_values(by=['category_id', 'size'], ascending=False)
    category_wise_popular_brand = category_wise_counts.drop_duplicates(subset=['category_id'])
    category_wise_popular_brand = category_wise_popular_brand.set_index('category_id')
    category_wise_popular_brand_submission = category_wise_popular_brand.ix[test['category_id'], 'brand_id'].fillna(
        41745).astype(int)
    category_wise_popular_brand_submission.to_csv('category_wise_popular_brand_submission.csv', index=False)




class Tokenizer(object):
    def __init__(self):
        self.tokenizer = word_tokenize
        self.stop_words = set(
        ['is', 'of', 'it', 'at', 'on', 'and', 'as', 'the', 'to', 'are', 'this', 'that', 'be', 'in',
          'an', 'or','any', 'all', 'am','you','we', '__NUMBER__', '__SERIAL__'])

    def __call__(self, text):
        text = text.lower()
        # replace special characters
        text = re.sub(r'[^a-z0-9\s/\\_\t,\-]', '', text,flags=re.IGNORECASE)
        text = re.sub(r'[/\\_\t,-]', ' ', text,flags=re.IGNORECASE)
        # replace numbers to reduce number of features
        text = re.sub(r'\b[0-9]+\b', ' __NUMBER__ ', text)
        # replace possible product/serial numbers
        text = re.sub(r'\b\w*\d+\w*\d?\b', ' __SERIAL__ ', text)

        tokens = [w for w in self.tokenizer(text) if (w not in self.stop_words and len(w)>1)]
        # only return first and last two tokens
        return tokens if len(tokens) <5 else tokens[:3] + tokens[-2:]

def learn_model_for_category(train_df,learner=MultinomialNB()):
    category = train_df.category_id.iloc[0]
    vectorizer = TfidfVectorizer(tokenizer=Tokenizer())
    estimators = [('transform', vectorizer), ('learner', learner)]
    pipe_line = Pipeline(estimators)
    pipe_line.fit(train_df['product_title'].values,train_df['brand_id'].astype(int))
    joblib.dump(pipe_line,'category_'+str(category)+'_model.clf')
    return True

def apply_model_for_category(test_df):
    category = test_df.category_id.iloc[0]
    try:
        learner = joblib.load('category_'+str(category)+'_model.clf')
        test_df.loc[test_df.index,'predicted_brand_id'] = learner.predict(test_df['product_title'].values)
        return test_df
    except Exception as e:
        print(e,test_df.shape)
        test_df.loc[test_df.index,'predicted_brand_id'] = -1
        return test_df

def learn_model_for_missing_category(train_df, test_df,learner=MultinomialNB()):
    test_vectorizer = TfidfVectorizer(tokenizer=Tokenizer())
    test_vectorizer.fit(test_df.product_title)
    category = "missing"
    vectorizer = TfidfVectorizer(tokenizer=Tokenizer(), vocabulary=test_vectorizer.vocabulary_)
    estimators = [('transform', vectorizer), ('learner', learner)]
    pipe_line = Pipeline(estimators)
    pipe_line.fit(train_df['product_title'].values,train_df['brand_id'].astype(int))
    joblib.dump(pipe_line,'category_'+str(category)+'_model.clf')
    return True

def apply_model_for_missing_category(test_df):
    category = 'missing'
    try:
        learner = joblib.load('category_'+str(category)+'_model.clf')
        test_df.loc[test_df.index,'predicted_brand_id'] = learner.predict(test_df['product_title'].values)
        return test_df
    except Exception as e:
        print(e,test_df.shape)
        test_df.loc[test_df.index,'predicted_brand_id'] = -1
        return test_df


def submission_3():
    cat_models = train.groupby('category_id').apply(learn_model_for_category)
    predictions = test.groupby('category_id').apply(apply_model_for_category)
    predictions.loc[predictions.index,'predicted_brand_id'] = predictions.predicted_brand_id.astype(int)
    unpredicted = predictions.query('predicted_brand_id == -1')

    tokenize = Tokenizer()
    test_vectorizer = TfidfVectorizer(tokenizer=tokenize())
    test_vectorizer.fit(unpredicted.product_title)

    vocab = test_vectorizer.vocabulary_.keys()
    missing_relevant_train = train['product_title'].apply(lambda x:vocab.isdisjoint(tokenize(x)))
    missing_train = train[~missing_relevant_train]
    # get top 5 popular categories
    mc = missing_train.category_id.value_counts()
    missing_train_df = missing_train[missing_train.category_id.isin(mc[:5].index)]
    learn_model_for_missing_category(missing_train_df, unpredicted)
    missing_predicted = apply_model_for_missing_category(unpredicted)
    predictions.loc[missing_predicted.index,'predicted_brand_id'] = missing_predicted.predicted_brand_id
    predictions.predicted_brand_id.to_csv('category_wise_mnb.csv',index=False)


def submission_4():
    clf = LogisticRegression()
    log_reg_learn = lambda df:learn_model_for_category(df,clone(clf))

    cat_models = train.groupby('category_id').apply(learn_model_for_category)
    predictions = test.groupby('category_id').apply(apply_model_for_category)
    predictions.loc[predictions.index,'predicted_brand_id'] = predictions.predicted_brand_id.astype(int)
    unpredicted = predictions.query('predicted_brand_id == -1')

    tokenize = Tokenizer()
    test_vectorizer = TfidfVectorizer(tokenizer=tokenize())
    test_vectorizer.fit(unpredicted.product_title)

    vocab = test_vectorizer.vocabulary_.keys()
    missing_relevant_train = train['product_title'].apply(lambda x:vocab.isdisjoint(tokenize(x)))
    missing_train = train[~missing_relevant_train]
    # get top 5 popular categories
    mc = missing_train.category_id.value_counts()
    missing_train_df = missing_train[missing_train.category_id.isin(mc[:5].index)]
    learn_model_for_missing_category(missing_train_df, unpredicted,clone(clf))
    missing_predicted = apply_model_for_missing_category(unpredicted)
    predictions.loc[missing_predicted.index,'predicted_brand_id'] = missing_predicted.predicted_brand_id
    predictions.predicted_brand_id.to_csv('category_wise_log_reg.csv',index=False)

def submission_5():
    clf = DecisionTreeClassifier()
    log_reg_learn = lambda df:learn_model_for_category(df,clone(clf))

    cat_models = train.groupby('category_id').apply(learn_model_for_category)
    predictions = test.groupby('category_id').apply(apply_model_for_category)
    predictions.loc[predictions.index,'predicted_brand_id'] = predictions.predicted_brand_id.astype(int)
    unpredicted = predictions.query('predicted_brand_id == -1')

    tokenize = Tokenizer()
    test_vectorizer = TfidfVectorizer(tokenizer=tokenize())
    test_vectorizer.fit(unpredicted.product_title)

    vocab = test_vectorizer.vocabulary_.keys()
    missing_relevant_train = train['product_title'].apply(lambda x:vocab.isdisjoint(tokenize(x)))
    missing_train = train[~missing_relevant_train]
    # get top 5 popular categories
    mc = missing_train.category_id.value_counts()
    missing_train_df = missing_train[missing_train.category_id.isin(mc[:5].index)]
    learn_model_for_missing_category(missing_train_df, unpredicted,clone(clf))
    missing_predicted = apply_model_for_missing_category(unpredicted)
    predictions.loc[missing_predicted.index,'predicted_brand_id'] = missing_predicted.predicted_brand_id
    predictions.predicted_brand_id.to_csv('category_wise_log_reg.csv',index=False)

def find_similar_brands():
    # to find duplicate brands

    # generate tf-idf of entire training
    vectorizer = TfidfVectorizer(tokenizer=Tokenizer())
    print_log("starting vectorizer fit_transform")
    sparse_title = vectorizer.fit_transform(train['product_title'])
    print_log("completed vectorizer fit_transform")

    # generate dummies of categories
    category_dict_vectorizer = DictVectorizer()
    print_log("starting sparse category")
    sparse_category = category_dict_vectorizer.fit_transform(train.category_id.astype(str).apply(lambda x: {x: 1}))
    print_log("completed sparse category")

    # stack tf-idf and categories
    joined_data = hstack([sparse_category, sparse_title], format='csr')

    # work at numpy level to get the advantage of dealing with sparse data
    # group by brand and aggregate tf-idf information
    brands = train['brand_id'].values
    unique_brands = np.unique(brands)
    joined_data_grouped = csr_matrix(np.zeros((1,joined_data.shape[1])))
    for brand in unique_brands:
        grp_sum = joined_data[brands == brand].sum(axis=0)
        joined_data_grouped = vstack([joined_data_grouped,grp_sum])

    joined_data_grouped = csr_matrix(joined_data_grouped)
    joined_data_grouped = joined_data_grouped[1:]

    # fit a LSH forest tree on aggregated data
    lshf = LSHForest(n_estimators=100,random_state=42)
    lshf.fit(joined_data_grouped)
    # query for five nearest neighbors for each brand
    distances, indices = lshf.kneighbors(joined_data_grouped, n_neighbors=5)
    return distances,indices

# generating submissions
submission_2()
submission_3()
submission_4()
submission_5()


