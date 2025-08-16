# RESTAURANT-REVIEW-PROCESSING-USING-NLP
🚀 Built an NLP pipeline to classify hotel reviews as positive/negative using NLTK + ML. Implemented KNN, Naive Bayes &amp; SVM, with SVM achieving best performance (~83% F1). Includes preprocessing, Bag of Words, and GridSearchCV tuning. Perfect for feedback analysis!

This project performs **sentiment analysis** on hotel/restaurant reviews using **Natural Language Processing (NLP)** and machine learning classifiers.
The dataset (`Restaurant_Reviews.tsv`) contains reviews labeled as **positive (1)** or **negative (0)**.

The pipeline involves:

1. **Text Preprocessing**
2. **Feature Extraction (Bag of Words)**
3. **Model Training & Evaluation**
4. **Hyperparameter Tuning**

---

## 📂 Dataset

* File: `Restaurant_Reviews.tsv`
* Columns:

  * `Review`: The review text.
  * `Liked`: Target label (1 = Positive, 0 = Negative).

---

## ⚙️ Requirements

Install dependencies before running:

```bash
pip install pandas numpy scikit-learn nltk openpyxl
```

Additionally, download NLTK stopwords:

```python
import nltk
nltk.download("stopwords")
```

---

## 🔄 Workflow

### 1. **Data Preprocessing**

* Removes punctuation, numbers, and special characters.
* Converts text to lowercase.
* Tokenizes and removes stopwords.
* Applies **PorterStemmer** to reduce words to their root forms.
* Example:

  ```
  "The food was amazing and loved it!" → "food amaz love"
  ```

### 2. **Feature Extraction**

* Uses **CountVectorizer** (Bag of Words).
* Keeps the **top 1500 most frequent words** (`max_features=1500`).

### 3. **Model Training**

Trains and compares three models:

* **K-Nearest Neighbors (KNN)**
* **Multinomial Naive Bayes (MNB)**
* **Linear Support Vector Classifier (SVC)**

### 4. **Evaluation Metrics**

Each model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

The best model is selected based on **F1-score**.

### 5. **Hyperparameter Tuning**

Uses **GridSearchCV** with 5-fold cross-validation:

* **KNN**

  ```python
  "n_neighbors": [2, 3, 7, 9, 11, 15],
  "metric": "minkowski",
  "p": [1, 2]
  ```
* **MultinomialNB**

  ```python
  "alpha": [0.1, 0.5, 1.0, 2.0]
  ```
* **SVC**

  ```python
  "C": [0.01, 0.1, 1, 10],
  "max_iter": [1000, 5000, 10000]
  ```

The tuned model’s performance is evaluated on the test set.

---

## 📊 Example Output

```
Knn -> ACC: 0.7250, PREC: 0.7000, REC:0.7400, F1: 0.7200
MNB -> ACC: 0.7800, PREC: 0.7700, REC:0.7900, F1: 0.7800
SVC -> ACC: 0.8200, PREC: 0.8100, REC:0.8300, F1: 0.8200

Best Model: SVC
Running Hyperparameter Tuning for SVC...
Best Parameters: {'C': 1, 'max_iter': 5000}
Best Cross-Validated F1: 0.8235
Tuned F1: 0.8350

Confusion Matrix:
[[78 12]
 [15 95]]
```

---

## 🚀 How to Run

1. Place the dataset `Restaurant_Reviews.tsv` in the specified folder.
2. Run the Python script:

```bash
python sentiment_analysis.py
```

3. Check the console output for metrics and the confusion matrix.

---

## 📌 Key Learnings

* Text preprocessing with **Regex + Stopwords + Stemming**.
* **Bag of Words** feature representation.
* Comparing classifiers (KNN, Naive Bayes, SVM).
* Importance of **hyperparameter tuning** for performance optimization.

---

