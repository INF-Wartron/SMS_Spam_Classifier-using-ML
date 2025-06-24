# SMS_Spam_Classifier-using-ML
This project is a **machine learning** application that classifies SMS messages as **spam** or **ham (legitimate)** using **Naive Bayes** and **TF-IDF vectorization**. It includes data preprocessing, model training, hyperparameter tuning, and evaluation with metrics like accuracy and F1-score.

---

## ✨ Features

* **Data Preprocessing**: Cleans and encodes labels (`ham` → 0, `spam` → 1)
* **TF-IDF Vectorization**: Converts raw text into feature vectors
* **Naive Bayes Classifier**: Efficient baseline model for text classification
* **GridSearchCV**: Optimizes model hyperparameters for the best performance
* **Evaluation Metrics**: Accuracy, F1-score, and detailed classification report

---

## 🛠️ Tech Stack

* **Python 3.x**
* **pandas** — for data handling
* **scikit-learn** — for model training, hyperparameter tuning, and evaluation

---

## 📂 Dataset

* The model uses a CSV file named `spam.csv` containing two columns:

  * `v1`: label (`ham` or `spam`)
  * `v2`: SMS message
* The file is loaded using `pd.read_csv()` with `latin-1` encoding.

---

## ⚙️ Usage Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/SMS-Spam-Classifier.git
   cd SMS-Spam-Classifier
   ```
2. Make sure you have all dependencies installed:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:

   ```bash
   python spam_classification.py
   ```
4. See the printed metrics for accuracy, F1-score, and the full classification report.

---

## 📊 Sample Output

```
--- Naive Bayes ---
Accuracy: 0.98
F1 Score: 0.95
Classification Report:
              precision    recall  f1-score   support
       0.0       0.99      0.99      0.99       ...
       1.0       0.94      0.91      0.93       ...
```

## 🎯 Future Improvements

* Implement other models like Logistic Regression or SVM
* Explore deep learning approaches (LSTM, BERT) for better accuracy
* Deploy as a web service or API for real-time predictions
