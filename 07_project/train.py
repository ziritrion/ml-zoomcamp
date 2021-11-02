import pandas as pd
#import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

import pickle


'''
Dataset loading
'''
original_columns = ['has_null', 'wave', 'gender', 'age', 'age_o', 'd_age', 'd_d_age', 'race', 'race_o', 'samerace', 'importance_same_race', 'importance_same_religion', 'd_importance_same_race', 'd_importance_same_religion', 'field', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'd_pref_o_attractive', 'd_pref_o_sincere', 'd_pref_o_intelligence', 'd_pref_o_funny', 'd_pref_o_ambitious', 'd_pref_o_shared_interests', 'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o', 'shared_interests_o', 'd_attractive_o', 'd_sinsere_o', 'd_intelligence_o', 'd_funny_o', 'd_ambitous_o', 'd_shared_interests_o', 'attractive_important', 'sincere_important', 'intellicence_important', 'funny_important', 'ambtition_important', 'shared_interests_important', 'd_attractive_important', 'd_sincere_important', 'd_intellicence_important', 'd_funny_important', 'd_ambtition_important', 'd_shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'd_attractive', 'd_sincere', 'd_intelligence', 'd_funny', 'd_ambition', 'attractive_partner', 'sincere_partner', 'intelligence_partner', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'd_attractive_partner', 'd_sincere_partner', 'd_intelligence_partner', 'd_funny_partner', 'd_ambition_partner', 'd_shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'd_sports', 'd_tvsports', 'd_exercise', 'd_dining', 'd_museums', 'd_art', 'd_hiking', 'd_gaming', 'd_clubbing', 'd_reading', 'd_tv', 'd_theater', 'd_movies', 'd_concerts', 'd_music', 'd_shopping', 'd_yoga', 'interests_correlate', 'd_interests_correlate', 'expected_happy_with_sd_people', 'expected_num_interested_in_me', 'expected_num_matches', 'd_expected_happy_with_sd_people', 'd_expected_num_interested_in_me', 'd_expected_num_matches', 'like', 'guess_prob_liked', 'd_like', 'd_guess_prob_liked', 'met', 'decision', 'decision_o', 'match']
columns_no_intervals = ['has_null', 'wave', 'gender', 'age', 'age_o', 'd_age', 'race', 'race_o', 'samerace', 'importance_same_race', 'importance_same_religion', 'field', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o', 'shared_interests_o', 'attractive_important', 'sincere_important', 'intellicence_important', 'funny_important', 'ambtition_important', 'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_partner', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate', 'expected_happy_with_sd_people', 'expected_num_interested_in_me', 'expected_num_matches', 'like', 'guess_prob_liked', 'met', 'match']

# Making a list of missing value types
missing_values = ["n/a", "na", "--", "?"]

df = pd.read_csv('speeddating.csv', na_values=missing_values, usecols=columns_no_intervals)

'''
Data cleaning
'''
categorical = ['gender', 'race', 'race_o', 'field']
numerical = ['has_null', 'wave', 'age', 'age_o', 'd_age', 'samerace', 'importance_same_race', 'importance_same_religion', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o', 'shared_interests_o', 'attractive_important', 'sincere_important', 'intellicence_important', 'funny_important', 'ambtition_important', 'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_partner', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate', 'expected_happy_with_sd_people', 'expected_num_interested_in_me', 'expected_num_matches', 'like', 'guess_prob_liked', 'met']

df['field'] = df['field'].str.lower()
df['field'] = df['field'].str.replace("'", "", regex=False)
df['field'] = df['field'].str.replace(" ", "_", regex=False)
df['field'] = df['field'].str.replace("[", "(", regex=False)
df['field'] = df['field'].str.replace("]", ")", regex=False)
df['field'] = df['field'].fillna('unknown')
df['field'] = df['field'].astype(str)

df['field'] = df['field'].str.replace('business-_mba', 'business_(mba)', regex=False)
df['field'] = df['field'].str.replace('business/law', 'business_(law)', regex=False)
df['field'] = df['field'].str.replace('business;_marketing', 'business_(marketing)', regex=False)
df['field'] = df['field'].str.replace('business;_media', 'business_(media)', regex=False)
df['field'] = df['field'].str.replace('business/_finance/_real_estate', 'business_(finance_&_real_estate)', regex=False)
df['field'] = df['field'].str.replace('creative_writing_-_nonfiction', 'creative_writing_(nonfiction)', regex=False)
df['field'] = df['field'].str.replace('climate-earth_and_environ._science', 'earth_and_environmental_science', regex=False)
df['field'] = df['field'].str.replace('electrical_engg.', 'electrical_engineering', regex=False)
df['field'] = df['field'].str.replace('finanace', 'finance', regex=False)
df['field'] = df['field'].str.replace('finance&economics', 'finance_&_economics', regex=False)
df['field'] = df['field'].str.replace('finance/economics', 'finance_&_economics', regex=False)
df['field'] = df['field'].str.replace('international_affairs/business', 'international_affairs_(business)', regex=False)
df['field'] = df['field'].str.replace('international_affairs/finance', 'international_affairs_(finance)', regex=False)
df['field'] = df['field'].str.replace('international_affairs/international_finance', 'international_affairs_(finance)', regex=False)
df['field'] = df['field'].str.replace('intrernational_affairs', 'international_affairs', regex=False)
df['field'] = df['field'].str.replace('master_in_public_administration', 'masters_in_public_administration', regex=False)
df['field'] = df['field'].str.replace('master_of_international_affairs', 'masters_in_international_affairs', regex=False)
df['field'] = df['field'].str.replace('math', 'mathematics', regex=False)
df['field'] = df['field'].str.replace('mfa__poetry', 'mfa_poetry', regex=False)
df['field'] = df['field'].str.replace('mfa_-film', 'mfa_film', regex=False)
#df['field'] = df['field'].str.replace('nan', 'unknown', regex=False)
df['field'] = df['field'].str.replace('nutritiron', 'nutrition', regex=False)
df['field'] = df['field'].str.replace('sipa_/_mia', 'masters_in_international_affairs', regex=False)
df['field'] = df['field'].str.replace('sipa-international_affairs', 'international_affairs', regex=False)
df['field'] = df['field'].str.replace('sociomedical_sciences-_school_of_public_health', 'sociomedical_sciences', regex=False)
df['field'] = df['field'].str.replace('speech_languahe_pathology', 'speech_pathology', regex=False)
df['field'] = df['field'].str.replace('speech_language_pathology', 'speech_pathology', regex=False)
df['field'] = df['field'].str.replace('stats', 'statistics', regex=False)
df['field'] = df['field'].str.replace('tc_(health_ed)', 'health_education', regex=False)

df['race'] = df['race'].str.lower()
df['race'] = df['race'].str.replace("'", "", regex=False)
df['race'] = df['race'].str.replace(" ", "_", regex=False)

df['race_o'] = df['race_o'].str.lower()
df['race_o'] = df['race_o'].str.replace("'", "", regex=False)
df['race_o'] = df['race_o'].str.replace(" ", "_", regex=False)

df.race = df.race.fillna('Unknown')
df.race_o = df.race_o.fillna('Unknown')

for n in numerical:
    df[n] = df[n].fillna(df[n].mean())

'''
Validation framework & one-hot encoding
'''
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
#df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.match.values
#y_test = df_test.match.values

del df_full_train['match']
#del df_test['match']

dv = DictVectorizer(sparse=False)

full_train_dict = df_full_train.to_dict(orient='records')
#test_dict = df_test.to_dict(orient='records')

X_full_train = dv.fit_transform(full_train_dict)
#X_test = dv.transform(test_dict)

features = dv.get_feature_names()
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
#dtest = xgb.DMatrix(X_test, feature_names=features)

'''
Training
'''
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    #'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dfulltrain, num_boost_round=50)

'''
Storing the model and DictVectorizer to files
'''
model_path = 'model.bin'
dv_path = 'dv.bin'

with open(model_path, 'wb') as f_out:
    pickle.dump(model, f_out)

with open(dv_path, 'wb') as f_out:
    pickle.dump(dv, f_out)