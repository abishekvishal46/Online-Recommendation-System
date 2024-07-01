from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Load datasets
data = pd.read_csv('Udemy.csv')
data_coursera = pd.read_csv("Coursera.csv")
data_edx = pd.read_csv("edx_courses.csv")

# Preprocess Coursera data
data_coursera['features'] = data_coursera['course'] + ' ' + data_coursera['reviewcount'].astype(str) + ' ' + \
                            data_coursera['level']
data_coursera['features'] = data_coursera['features'].fillna('')
vectorizer_coursera = TfidfVectorizer(stop_words='english')
feature_matrix_coursera = vectorizer_coursera.fit_transform(data_coursera['features'])

# Preprocess edX data
text_features = ['title', 'summary', 'instructors', 'Level', 'price', 'course_url']
data_edx['combined_text'] = data_edx[text_features].astype(str).apply(lambda x: ' '.join(x), axis=1)
vectorizer_edx = TfidfVectorizer(stop_words='english')
feature_matrix_edx = vectorizer_edx.fit_transform(data_edx['combined_text'])


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        title = request.form.get("title")
        if title:
            # Process Udemy recommendations
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            data['description'] = data['description'].fillna('')
            tfidf_matrix = tfidf_vectorizer.fit_transform(data['description'])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            matches = data[data['title'].str.contains(title, case=False)]
            if not matches.empty:
                idx1 = matches.index[0]
                sim_scores = list(enumerate(cosine_sim[idx1]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:11]
                course_indices = [score[0] for score in sim_scores]
                recommended_courses = data.iloc[course_indices]
                recommended_courses = recommended_courses[
                    recommended_courses['description'].str.contains(title, case=False)]
            else:
                recommended_courses = None

            # Process Coursera recommendations
            input_features = [title]
            input_features_vector = vectorizer_coursera.transform(input_features)
            similarity_scores = cosine_similarity(feature_matrix_coursera, input_features_vector)
            similar_indices = similarity_scores.argsort(axis=0)[-10 - 1:-1][::-1]
            top_recommendations = [data_coursera.iloc[idx[0]] for idx in similar_indices]

            # Process edX recommendations
            input_features_edx = [title]
            input_features_edx_vector = vectorizer_edx.transform(input_features_edx)
            similarity_scores_edx = cosine_similarity(feature_matrix_edx, input_features_edx_vector)
            similar_indices_edx = similarity_scores_edx.argsort(axis=0)[-10 - 1:-1][::-1]
            top_edx = [data_edx.iloc[idx[0]] for idx in similar_indices_edx]

            return render_template("index.html",
                                   recommended_courses_udemy=recommended_courses,
                                   recommended_courses_coursera=enumerate(top_recommendations),
                                   recommended_courses_edx=enumerate(top_edx))
        else:
            return render_template("index.html",
                                   error="Title parameter is missing.",
                                   recommended_courses_udemy=None,
                                   recommended_courses_coursera=None,
                                   recommended_courses_edx=None)

    return render_template("index.html",
                           recommended_courses_udemy=None,
                           recommended_courses_coursera=None,
                           recommended_courses_edx=None)


if __name__ == '__main__':
    app.run(debug=True)
