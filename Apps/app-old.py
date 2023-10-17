import streamlit as st
import pickle
import pandas as pd


def recommend(movie):
    movie_index = movies[movies['Title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommend_movies = []
    recommend_movies_image = []
    recommended_dish_recipie = []
    ingredients = []
    for i in movies_list:
        movie_id = i[0]
        # Fetch poster from API
        recommend_movies.append(movies["Title"][i[0]])
        recommend_movies_image.append((movies["Image_Name"][i[0]]))
        recommended_dish_recipie.append((movies["Instructions"][i[0]]))
        ingredients.append((movies["Cleaned_Ingredients"][i[0]]))

    return recommend_movies, recommend_movies_image, recommended_dish_recipie, ingredients


movies_dict = pickle.load(open('Models/DishBasedRecom.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('Models/DishBasedRecomSimilarity.pkl', 'rb'))

st.title('Recipe Recommender System')

selected_movie_name = st.selectbox("Enter the Dish you have currently liked", movies['Title'].values)

if st.button('Recommend'):
    name_recommendations, image_recommendations, recipie, ingredient = recommend(selected_movie_name)
    st.write(f"Recommending top 5 Dishes similar to {selected_movie_name}...")
    for i, j, k, l in zip(name_recommendations, image_recommendations, recipie, ingredient):
        st.header(i, divider='rainbow')
        st.image(f"Food Images/Food Images/{j}.jpg")
        st.subheader("Ingredients: ", divider='rainbow')
        st.write(l)
        st.subheader("Recipie: ", divider='rainbow')
        st.write(k)
