import numpy as np
import pandas as pd
import ast

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

credits.head(1)['cast']

movies = movies.merge(cast,on='title')

movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast_x','cast_y', 'crew_x', 'crew_y']]
movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()
movies.iloc[0].genres

def convert(obj):
  L = []
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

movies['keywords']=movies['keywords'].apply(convert)

def convert3 (obj):
  L = []
  counter = 0
  for i in ast.literal_eval(obj):
    if counter !=3:
      L.append(i['name'])
      counter+=1
    else:
      break
  return L

def fetch_title(obj):
  L = []
  for i in ast.literal_eval(obj):
    if i['job']=='title':
      L.append(i['name'])
      break
  return L

movies['id'].apply(fetch_title)

movies['genres']= movies['genres'].apply(lambda x:[i.replace(" "," ") for i in x])
movies['keywords']= movies['keywords'].apply(lambda x:[i.replace(" "," ") for i in x])

new_df = movies[['id', 'title', 'tagline']]

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features= 5000, stop_words= 'english')

vectors= cv.fit_transform(new_df['title']).toarray()

from google.colab import drive
drive.mount('/content/drive')

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

cv.get_feature_names()

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
  y = []
  for i in text.split():
    y.append(ps.stem(i))

  return "".join(y)

ps.stem('loving')

new_df['title']= new_df['title'].apply(stem)

sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x:x[1])[1:6]

def recommend(movie):
  movie_index= new_df[new_df['title']==movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1]) [1:6]

  for i in movies_list:

    print(new_df.iloc[i[0]].title)


recommend('Batman Begins')




