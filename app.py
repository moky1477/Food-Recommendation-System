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

# Load datasets
ingredient_data = pd.read_csv("Data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
dish_data = pd.read_csv("Data/labeled_dataset.csv")

# Preprocess data
ingredient_data.drop('Unnamed: 0', axis=1, inplace=True)
dish_data["Cleaned_Ingredients_New"] = dish_data["Ingredients"].apply(lambda text: re.sub(r'[^a-zA-Z\s,]', '', text.lower()))

# Load pre-trained models
movies_dict = pickle.load(open('Models/DishBasedRecom.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('Models/DishBasedRecomSimilarity.pkl', 'rb'))

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
ingredient_vectors = cv.fit_transform(ingredient_data["Cleaned_Ingredients"]).toarray()
dish_vectors = cv.fit_transform(dish_data["Cleaned_Ingredients_New"])

# Text processing functions
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s,]', '', text.lower())
    words = word_tokenize(text)
    return ' '.join(words)

def stem(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(i) for i in text.split()])

@st.cache_data
def recommend_dishes_by_ingredients(input_ingredients, dataset, vectors, vectorizer, top_n=5):
    cleaned_input = stem(clean_text(input_ingredients))
    input_vector = vectorizer.transform([cleaned_input]).toarray()
    similarities = cosine_similarity(input_vector, vectors)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    return dataset.iloc[top_indices][["Title", "Instructions", "Image_Name", "Cleaned_Ingredients"]]

@st.cache_data
def recommend_dishes_by_liked_dish(selected_dish, dataset, vectors, vectorizer, top_n=5):
    dish_index = dataset[dataset['Title'] == selected_dish].index[0]
    similarities = cosine_similarity(vectors[dish_index], vectors)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    return dataset.iloc[top_indices][["Title", "Instructions", "Image_Name", "Cleaned_Ingredients"]]

# Streamlit App
st.title('🍽️ Recipe Recommender System')
option = st.radio("Choose Recommendation Type:", ["Ingredient-Based", "Dish-Based"])

if option == "Ingredient-Based":
    user_input = st.text_area("Enter the ingredients you like (comma-separated)", "")
    if st.button('Recommend'):
        if user_input:
            recommended_dishes = recommend_dishes_by_ingredients(user_input, ingredient_data, ingredient_vectors, cv)
            st.write("### Recommended Dishes")
            for _, row in recommended_dishes.iterrows():
                st.header(row['Title'])
                image_path = f"Food Images/Food Images/{row['Image_Name']}.jpg"
                img = Image.open(image_path)
                st.image(img, caption=row['Title'])
                st.subheader("Ingredients:")
                st.write(row['Cleaned_Ingredients'])
                st.subheader("Recipe:")
                st.write(row['Instructions'])
        else:
            st.write("Please enter ingredients.")

elif option == "Dish-Based":
    selected_dish = st.selectbox("Select a dish you like:", dish_data['Title'].values)
    veg_or_non_veg = st.radio("Select Dish Type:", ["Veg", "Non-Veg"])
    if st.button('Recommend'):
        filtered_data = dish_data[dish_data['dish_type'].str.lower() == veg_or_non_veg.lower()]
        recommended_dishes = recommend_dishes_by_liked_dish(selected_dish, filtered_data, dish_vectors, cv)
        st.write(f"### Recommended Dishes Similar to {selected_dish}")
        for _, row in recommended_dishes.iterrows():
            st.header(row['Title'])
            image_path = f"Food Images/Food Images/{row['Image_Name']}.jpg"
            img = Image.open(image_path)
            st.image(img, caption=row['Title'])
            st.subheader("Ingredients:")
            st.write(row['Cleaned_Ingredients'])
            st.subheader("Recipe:")
            st.write(row['Instructions'])
