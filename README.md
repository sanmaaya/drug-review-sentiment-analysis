# drug-review-sentiment-analysis
NLP and ML-based sentiment classification of drug reviews with Power BI dashboard visualization.

# 💊 Drug Review Sentiment Analysis

A Natural Language Processing (NLP) and Machine Learning project to classify user sentiments in real-world drug reviews as **Positive**, **Negative**, or **Neutral**, with results visualized through an interactive **Power BI dashboard**.

---

## 🔍 Features

- ✅ Text cleaning and preprocessing
- ✅ TF-IDF vectorization for feature extraction
- ✅ Sentiment classification using Logistic Regression
- ✅ Evaluation metrics: accuracy, confusion matrix, classification report
- ✅ Interactive Power BI dashboard with visual insights

---

## 📁 Files Included

| File                         | Description                                 |
|------------------------------|---------------------------------------------|
| `data_cleaning.py`           | Clean raw review data (punctuation, case)   |
| `nlp.py`                     | Tokenization, TF-IDF vectorization          |
| `feature_engineering.py`     | Train-test split and data saving            |
| `model_evaluator.py`         | Train Logistic Regression and evaluate      |
| `model_predictions.csv`      | CSV file with predicted sentiments          |
| `metrics.txt`                | Accuracy and classification report          |
| `plots/`                     | Confusion matrix and performance graphs     |
| `sentiment_analysis_project.pbix` | Power BI dashboard file               |

---

## 📊 Dashboard Features (Power BI)

- 📌 **Sentiment Distribution**  
- 📌 **Review Trends by Year**  
- 📌 **Top Conditions Reviewed**  
- 📌 **Review Length by Sentiment**  
- 📌 **Rating Distribution**

---

## 🧠 Project Workflow

1. **Data Cleaning** – Removed punctuation, converted to lowercase, formatted date columns  
2. **NLP Preprocessing** – Used TF-IDF to convert text into numerical vectors  
3. **Feature Engineering** – Split dataset into train and test sets, saved for reuse  
4. **Model Training** – Trained Logistic Regression on vectorized data  
5. **Evaluation** – Calculated accuracy, classification report, and confusion matrix  
6. **Visualization** – Built Power BI dashboard with model output and insights  

---

## ⚙️ How to Run the Project

Make sure you have Python and required libraries installed.

```bash
python data_cleaning.py
python nlp.py
python feature_engineering.py
python model_evaluator.py
