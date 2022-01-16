import streamlit as st
import pandas as pd
import numpy as np
import os
import json

import warnings
from numpy.linalg import norm
from operator import itemgetter
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("Welcome to FoodPie!")
st.sidebar.title('FoodPie!')

FP_PATH = 'food_profiles.npz.npy'
U_PATH = 'users.csv'

dic = {}
with open("dic.json", "r") as outfile:
    dic = json.load(outfile)
#st.write(dic)
def load_data():
    food_profiles = np.load(FP_PATH)
    users = pd.read_csv(U_PATH).drop(['Unnamed: 0'], axis=1)
    users['date'] = pd.to_datetime(users['date']).dt.date
    users.sort_values(['date'], ascending=False, inplace=True)
    users.rename(columns={'recipe_id':'item_id'}, inplace=True)
    return users, food_profiles

def cosine_sim(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))

def sort_tuple(data, sort_key, descending=True):
    return sorted(data, key=itemgetter(sort_key), reverse=descending)

def recom(user, food_profiles, raw_b, food_ids_list):
    count =0
    taste_profile = np.zeros(food_profiles[:, 0].shape)

    for i in raw_b[raw_b['user_id']==user]['item_id'].unique().tolist():
        index = food_ids_list.index(i)
        taste_profile += food_profiles[:, index]
        count += 1

    if count > 0:
        taste_profile = taste_profile/count
    #print(taste_profile)
    _scores = []
    for i in range(0, len(food_ids_list)):
        similarity = cosine_sim(taste_profile, food_profiles[:, i])
        _scores.append((str(food_ids_list[i]), similarity))
    _scores = sort_tuple(data=_scores, sort_key=1, descending=True)

    _final_list = []
    for _s in _scores:
        _final_list.append({'item_id': _s[0], 'score': _s[1].round(3)})
    data = pd.DataFrame(_final_list)[:10]
    data['item'] = list(map(lambda x: dic.get((x))[0], list(data['item_id'])))
    return data



user_data, food_profiles = load_data()
food_ids_list = user_data['item_id'].unique().tolist()
user_tup = tuple(user_data['user_id'].unique())
user = st.sidebar.selectbox('UserID', user_tup)
login = st.sidebar.button('Login')
if login:
    st.info(f'Logged in as "{user}"')
    u = user_data[user_data['user_id']==user]
    st.write('')
    my_expander = st.expander("Show user data")
    with my_expander:
        st.write(u.head(20))
    st.write('')

    st.subheader(f'Serving Top 10 Recommendations to user {user}')
    recoms = recom(user, food_profiles, user_data, food_ids_list)
    st.write(recoms)
  
