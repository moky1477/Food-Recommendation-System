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
data = pd.read_csv("Data/labeled_dataset.csv")

# Load pre-trained models and data - Not really IMP here
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

    # Get the indices of dishes sorted by similarity in descending order
    top_indices = np.argsort(similarities[0])[::-1]

    # Filter out the indices of dishes that are not in the filtered dataset
    top_indices = [idx for idx in top_indices if idx in dataset.index]

    # Take the top n recommendations (or fewer if there are not enough)
    top_indices = top_indices[:top_n]

    top_dishes = dataset.loc[top_indices, ["Title", "Instructions"]]
    return top_dishes


# Streamlit app
st.title('Recipe Recommender System')

selected_dish = st.selectbox("Enter the Dish you have currently liked", data['Title'].values)

veg_or_non_veg = st.radio("Select Dish Type:", ["Veg", "Non-Veg"])

if st.button('Recommend'):
    if veg_or_non_veg == "Veg":
        # Filter the dataset to include only veg dishes
        filtered_data = data[data['dish_type'] == 'veg']
    else:
        # Filter the dataset to include only non-veg dishes
        filtered_data = data[data['dish_type'] == 'non-veg']

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
