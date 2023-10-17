# Initial and 1st app made, takes time to execute but working perfectly fine
# Recommends Dish on Liked Dish

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
import IPython.display as display

# Load your dataset
data = pd.read_csv("Data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
data.drop('Unnamed: 0', inplace=True, axis=1)

# Load pre-trained models and data
movies_dict = pickle.load(open('Models/DishBasedRecom.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('Models/DishBasedRecomSimilarity.pkl', 'rb'))


# Clean and preprocess data (similar to your original code)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s,]', '', text)
    text = re.sub(r'\b(?:lb|tsp|tbsp|total|cup|cups)\b', '', text)
    words = word_tokenize(text)
    cleaned_text = ' '.join(words)
    return cleaned_text


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


ps = PorterStemmer()
data["Cleaned_Ingredients_New"] = data["Ingredients"].apply(clean_text)
data["Cleaned_Ingredients_New"] = data["Cleaned_Ingredients_New"].apply(stem)
cv = CountVectorizer(max_features=5000, stop_words='english')
ingredient_vectors = cv.fit_transform(data["Cleaned_Ingredients_New"]).toarray()


# Define recommendation function
def recommend_dishes(input_ingredients, dataset, vectors, vectorizer, top_n=5):
    cleaned_input = clean_text(input_ingredients)
    stemmed_input = stem(cleaned_input)
    input_vector = vectorizer.transform([stemmed_input]).toarray()
    similarities = cosine_similarity(input_vector, vectors)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    top_dishes = dataset.iloc[top_indices][["Title", "Instructions"]]
    return top_dishes


# Streamlit app
st.title('Recipe Recommender System')

selected_dish = st.selectbox("Enter the Dish you have currently liked", movies['Title'].values)

if st.button('Recommend'):
    recommended_dishes = recommend_dishes(selected_dish, data, ingredient_vectors, cv)
    st.title(f'Recommended Dishes for {selected_dish}')

    st.write(f"Recommending top 5 Dishes")
    for i, (dish, instructions) in enumerate(recommended_dishes.iterrows(), start=1):
        title = instructions['Title']
        instructions = instructions['Instructions']
        image_name = data['Image_Name'][dish]
        original_index = dish
        ingredients = data['Cleaned_Ingredients'][dish]

        # Load and display the image
        image_path = f"Food Images/Food Images/{image_name}.jpg"
        img = Image.open(image_path)
        st.header(title, divider='rainbow')
        st.image(img, caption=f"{title}")
        st.subheader("Ingredients: ", divider='rainbow')
        st.write(ingredients)
        st.subheader("Recipie: ", divider='rainbow')
        st.write(instructions)


