# 🚀 Spam Classification & Generation Project

A complete NLP pipeline that detects spam messages and enhances the dataset using a character-level generative LSTM model. This project combines traditional machine learning with deep learning approaches and includes data augmentation via spam-like sentence generation.                              

---

## 🔍 Project Highlights   

- **Task**: Classify SMS messages into `spam` or `ham`.   
- **Dataset**: `SpamVsHam.tsv` - real-world SMS spam dataset.   
- **Models**:   
  - Traditional ML: `GaussianNB`, `MultinomialNB` with `TF-IDF` / `Word2Vec`   
  - Deep Learning: Custom `LSTM` models with `PyTorch`   
- **Data Augmentation**: Trained LSTM generator to create synthetic SPAM messages   
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC Curve   

---

## 📂 Project Structure   

### 1. Data Preprocessing   

- Load and inspect `SpamVsHam.tsv`.   
- Clean and split into `spam` and `ham` subsets.   
- Map labels and preprocess text.   

### 2. Feature Extraction   

- **Word2Vec** for embedding each word into vector space.   
- **TF-IDF** for bag-of-words representation.   

---

### 3. ML Model: Naive Bayes   

- Model 1: `GaussianNB` for Word2Vec embeddings.   
- Model 2: `MultinomialNB` for TF-IDF vectors.   
- Training/Validation splitting and model evaluation.   

> **Key Result**: Traditional NB models work well for text classification, but may underperform with continuous embeddings like Word2Vec.   

---

### 4. DL Model: LSTM (Classifier)   

- Inputs: Word2Vec or TF-IDF vectors as sequences.   
- Architecture:   
  - LSTM layer (1 or more)   
  - Dropout   
  - Fully connected output   
- Loss: Binary Cross Entropy   
- Optimizer: Adam   
- Output: Probability of message being `spam`   

> **Training**: Tracked loss and accuracy for both training and validation.   

---

### 5. LSTM Spam Generator (Text Generation)   

- Built a character-level LSTM model trained on only `spam` messages.   
- Inputs: sequences of characters (length=50)   
- Output: next character prediction   
- Generation: feed initial spam sentence (seed), generate new spam-like sentences   

> **Used for Data Augmentation**   

---

### 6. Data Augmentation   

- Generated 100 spam-like synthetic texts using LSTM model.   
- Merged with original dataset.   
- Re-trained ML and DL models with expanded dataset.   
- Compared performance before and after augmentation.   

> **Result**: Boosted classification robustness by exposing the models to diverse spam patterns.   

---

## 🔄 Evaluation Metrics   

- Accuracy   
- Precision / Recall / F1   
- Confusion Matrix   
- ROC Curve (AUC)   

---

## 🔧 Technologies Used   

- Python   
- Pandas, NumPy   
- Scikit-learn   
- Gensim (Word2Vec)   
- PyTorch   
- Matplotlib, Seaborn   

---

## 💼 Usage   

```bash
# Clone repository
$ git clone https://github.com/your-username/spam-classification-lstm.git

# Install dependencies
$ pip install -r requirements.txt

# Run Notebook
Open and execute main.ipynb in Jupyter Lab or Jupyter Notebook
```

---

## 🚀 Future Improvements   

- Replace Word2Vec with BERT or other contextual embeddings   
- Hyperparameter tuning   
- Expand generative model to include ham messages   
- Deploy as REST API or interactive web app   

---

## 🌟 Credits   

Dataset from public SMS spam corpus. Project structure inspired by end-to-end NLP classification workflows.   

---

## 🌐 License   

MIT License
