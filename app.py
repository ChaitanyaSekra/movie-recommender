from fastapi import FastAPI, Request, Form, Depends, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import pandas as pd
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import DictCursor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class MovieDatabase:
    def __init__(self, dbname: str, user: str, password: str, host: str = 'localhost', port: int = 5432):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.conn.autocommit = True

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        with self.conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(query, params or ())
            if cursor.description:
                return [dict(row) for row in cursor.fetchall()]
            return []

db = MovieDatabase('IMDb', 'postgres', 'root')

feature_options = {
    "all": ['genre', 'director', 'stars'],
    "genre": ['genre'],
    "director": ['director'],
    "stars": ['stars']
}

def get_movies_df() -> pd.DataFrame:
    query = "SELECT title, genre, director, stars, rating FROM da_movies1;"
    result = db.execute_query(query)
    df = pd.DataFrame(result)

    if df.empty:
        raise ValueError("No data returned from the database!")

    df.columns = df.columns.astype(str).str.strip().str.lower()
    for col in ['title', 'genre', 'director', 'stars']:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: '{col}'")
        df[col] = df[col].astype(str).fillna('')
    return df

def build_combined_column(df, features):
    return df[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

def compute_similarity_matrix(df, features):
    combined_data = build_combined_column(df, features)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined_data)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, df, cosine_sim, threshold=0.3):
    try:
        idx = df[df['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    recommendations = []
    for i, score in sim_scores:
        if score >= threshold:
            recommendations.append({
                'title': df.iloc[i]['title'],
                'genre': df.iloc[i]['genre'],
                'director': df.iloc[i]['director'],
                'stars': df.iloc[i]['stars'],
                'rating': df.iloc[i]['rating'],
                'similarity': round(score, 2)
            })
    return recommendations

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/recommend")
@app.post("/recommend")
async def recommend(request: Request):
    df = get_movies_df()
    movies = sorted(df['title'].dropna().unique().tolist())

    form = await request.form() if request.method == "POST" else None
    selected_movie = form.get("movie") if form else ""
    selected_feature = form.get("feature") if form else "all"
    recommendations = []

    if form:
        cosine_sim = compute_similarity_matrix(df, feature_options[selected_feature])
        recommendations = get_recommendations(selected_movie, df, cosine_sim)

    return templates.TemplateResponse("recommend.html", {
        "request": request,
        "movies": movies,
        "recommendations": recommendations,
        "selected_movie": selected_movie,
        "selected_feature": selected_feature,
        "feature_options": feature_options
    })

@app.get("/rate")
@app.post("/rate")
async def rate_movie(request: Request):
    df = get_movies_df()
    movies = sorted(df['title'].unique().tolist())
    message = None  # Initialize message

    if request.method == "POST":
        form = await request.form()
        movie_title = form.get("movie")
        email = form.get("email")
        rating = form.get("rating")

        # Validate rating input
        try:
            rating = float(rating)
            if not (0 <= rating <= 10):
                raise ValueError
        except ValueError:
            message = "⚠️ Rating must be a number between 0 and 10."
            return templates.TemplateResponse("rate.html", {"request": request, "movies": movies, "message": message})

        # Check user email existence
        user_query = "SELECT user_id FROM users WHERE email = %s"
        user_data = db.execute_query(user_query, (email,))
        if not user_data:
            message = "❌ Email does not exist. Please register first."
            return templates.TemplateResponse("rate.html", {"request": request, "movies": movies, "message": message})

        user_id = user_data[0]['user_id']

        # Check movie existence
        movie_query = "SELECT movie_id FROM da_movies1 WHERE title = %s LIMIT 1"
        movie_data = db.execute_query(movie_query, (movie_title,))
        if not movie_data:
            message = "❌ Selected movie does not exist."
            return templates.TemplateResponse("rate.html", {"request": request, "movies": movies, "message": message})

        movie_id = movie_data[0]['movie_id']

        # Insert or update rating
        insert_query = """
            INSERT INTO ratings (user_id, movie_id, rating)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, movie_id) DO UPDATE SET rating = EXCLUDED.rating
        """
        try:
            db.execute_query(insert_query, (user_id, movie_id, rating))
            message = "✅ Rating submitted successfully!"
        except Exception as e:
            print(f"Database error: {str(e)}")
            message = "❌ Failed to submit rating. Please try again."

    return templates.TemplateResponse("rate.html", {"request": request, "movies": movies, "message": message})

@app.get("/user")
@app.post("/user")
async def user(request: Request):
    message = None  # ✅ Define message at the top for both GET and POST

    if request.method == "POST":
        form = await request.form()
        username = form.get("username")
        email = form.get("email")

        if not username or not email:
            message = "⚠️ Username and Email are required."
            return templates.TemplateResponse("user.html", {"request": request, "message": message})

        check_query = "SELECT email FROM users WHERE email = %s"
        existing = db.execute_query(check_query, (email,))

        if existing:
            message = "⚠️ Email already exists."
            return templates.TemplateResponse("user.html", {"request": request, "message": message})

        insert_query = "INSERT INTO users (username, email) VALUES (%s, %s)"
        try:
            db.execute_query(insert_query, (username, email))
            message = "✅ User created successfully!"
        except Exception as e:
            print(f"Database error: {str(e)}")
            message = "❌ User not created. Please try again."

    # Return the form with optional message (works for both GET and POST)
    return templates.TemplateResponse("user.html", {"request": request, "message": message})



#     python -m uvicorn app:app --reload

