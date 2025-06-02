from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("Movies_data.csv", encoding='utf-8-sig')
df = df.fillna('')
df.columns = df.columns.str.strip()

feature_options = {
    "all": ['genre', 'director', 'stars'],
    "genre": ['genre'],
    "director": ['director'],
    "stars": ['stars']
}

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
                'similarity': round(score, 4)
            })
    return recommendations

@app.route("/", methods=["GET", "POST"])
def index():
    movies = df['title'].dropna().tolist()  # Drop NaNs if any
    recommendations = []
    selected_movie = ""
    selected_feature = "all"

    if request.method == "POST":
        selected_movie = request.form["movie"]
        selected_feature = request.form["feature"]
        cosine_sim = compute_similarity_matrix(df, feature_options[selected_feature])
        recommendations = get_recommendations(selected_movie, df, cosine_sim)

    return render_template(
    "index.html",
    movies=movies,
    recommendations=recommendations,
    selected_movie=selected_movie,
    selected_feature=selected_feature,
    feature_options=feature_options
)



if __name__ == "__main__":
    app.run(debug=True)
