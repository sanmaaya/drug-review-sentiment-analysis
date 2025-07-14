# drug-review-sentiment-analysis
NLP and ML-based sentiment classification of drug reviews with Power BI dashboard visualization.
💊 Drug Review Sentiment Analysis
This is a Natural Language Processing (NLP) project where I analyzed real-world drug reviews to predict user sentiment as Positive, Negative, or Neutral using Machine Learning. The final results were visualized through an interactive Power BI dashboard.

🔍 Features
Text cleaning and preprocessing

TF-IDF vectorization

Sentiment classification using Logistic Regression

Evaluation metrics: accuracy, confusion matrix, classification report

Interactive Power BI dashboard with visual insights

📁 Files Included
data_cleaning.py – Clean raw review data

nlp.py – Text processing and TF-IDF vectorization

feature_engineering.py – Train-test split and dataset saving

model_evaluator.py – Train and evaluate the model

sentiment_analysis_project.pbix – Power BI dashboard

model_predictions.csv, metrics.txt, and evaluation plots

📊 Dashboard Features
Sentiment Distribution

Review Trends by Year

Top Conditions Reviewed

Review Length by Sentiment

Rating Distribution

🧠 Project Workflow
Data Cleaning – Removed punctuation, lowercase text, formatted dates

NLP Preprocessing – Applied TF-IDF for vectorizing review text

Feature Engineering – Performed train-test split and saved processed data

Model Training – Trained Logistic Regression classifier

Evaluation – Generated accuracy, classification report, and confusion matrix

Visualization – Built a Power BI dashboard using predictions and trends

⚙️ How to Run
python data_cleaning.py
python nlp.py
python feature_engineering.py
python model_evaluator.py

🙌 Acknowledgement
This project was originally a collaborative academic assignment. This version reflects my own individual implementation and visualization.
Grateful to my project guide Sandeep Kaur and peers for inspiration.
