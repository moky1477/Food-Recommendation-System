# Caching implemented for faster running of model, veg-non veg button working, very good app,
# best working app untill 10-10-2023
# 10-10 Recommends Dish on Liked Dish

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load your dataset
data = pd.read_csv("Data/labeled_dataset.csv")

# Load pre-trained models and data - Not really IMP here
movies_dict = pickle.load(open('Models/DishBasedRecom.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('Models/DishBasedRecomSimilarity.pkl', 'rb'))


# Clean and preprocess data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s,]', '', text)
    text = re.sub(r'\b(?:lb|tsp|tbsp|total|cup|cups)\b', '', text)
    return text


data["Cleaned_Ingredients_New"] = data["Ingredients"].apply(preprocess_text)

cv = CountVectorizer(max_features=5000, stop_words='english')
ingredient_vectors = cv.fit_transform(data["Cleaned_Ingredients_New"])


# Define recommendation function
@st.cache_data
def recommend_dishes(input_ingredients, dataset, _vectors, _vectorizer, top_n=5):
    cleaned_input = preprocess_text(input_ingredients)
    input_vector = _vectorizer.transform([cleaned_input])
    similarities = cosine_similarity(input_vector, _vectors)

    top_indices = np.argsort(similarities[0])[::-1]
    top_indices = [idx for idx in top_indices if idx in dataset.index]
    top_indices = top_indices[:top_n]

    top_dishes = dataset.loc[top_indices, ["Title", "Instructions"]]
    return top_dishes


# Streamlit app
st.title('Recipe Recommender System')

selected_dish = st.selectbox("Enter the Dish you have currently liked", data['Title'].values)

veg_or_non_veg = st.radio("Select Dish Type:", ["Veg", "Non-Veg"])

if st.button('Recommend'):
    if veg_or_non_veg == "Veg":
        filtered_data = data[data['dish_type'] == 'veg']
    else:
        filtered_data = data

    recommended_dishes = recommend_dishes(selected_dish, filtered_data, ingredient_vectors, cv)
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
        st.subheader("Recipe: ", divider='rainbow')
        st.write(instructions)
