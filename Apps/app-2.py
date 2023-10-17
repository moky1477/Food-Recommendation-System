# Recommends Dish on given ingredients as input

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load your dataset
data = pd.read_csv("Data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
data.drop('Unnamed: 0', inplace=True, axis=1)

# Load pre-trained models and data
cv = CountVectorizer(max_features=5000, stop_words='english')
ingredient_vectors = cv.fit_transform(data["Cleaned_Ingredients"]).toarray()


# Define recommendation function
def recommend_dishes(input_ingredients, dataset, vectors, vectorizer, top_n=5):
    cleaned_input = clean_text(input_ingredients)
    stemmed_input = stem(cleaned_input)
    input_vector = vectorizer.transform([stemmed_input]).toarray()
    similarities = cosine_similarity(input_vector, vectors)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    top_dishes = dataset.iloc[top_indices][["Title", "Instructions"]]
    return top_dishes


# Text cleaning and stemming functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s,]', '', text)
    text = re.sub(r'\b(?:lb|tsp|tbsp|total|cup|cups)\b', '', text)
    words = word_tokenize(text)
    cleaned_text = ' '.join(words)
    return cleaned_text


def stem(text):
    ps = PorterStemmer()
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# Streamlit app
st.title('Recipe Recommender System')

user_input = st.text_area("Enter the ingredients you like (comma-separated)", "")

if st.button('Recommend'):
    if user_input:
        recommended_dishes = recommend_dishes(user_input, data, ingredient_vectors, cv)
        st.write(f"Recommending top 5 Dishes based on the ingredients you like...")
        for i, (dish, instructions) in enumerate(recommended_dishes.iterrows(), start=1):
            title = instructions['Title']
            instructions = instructions['Instructions']
            image_name = data['Image_Name'][dish]
            original_index = dish
            ingredients = data['Cleaned_Ingredients'][original_index]

            # Load and display the image
            image_path = f"Food Images/Food Images/{image_name}.jpg"
            img = Image.open(image_path)
            st.header(title, divider='rainbow')
            st.image(img, caption=f"{title}")
            st.subheader("Ingredients: ", divider='rainbow')
            st.write(ingredients)
            st.subheader("Recipie: ", divider='rainbow')
            st.write(instructions)
    else:
        st.write("Please enter the ingredients you like.")
