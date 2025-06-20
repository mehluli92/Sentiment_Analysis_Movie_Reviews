Sentiment Analysis on IMDB Movie Reviews
Project Overview
This project implements a sentiment analysis model using a Long Short-Term Memory (LSTM) neural network to classify movie reviews from the IMDB dataset as positive or negative. The goal is to enable stakeholders, such as movie studios or streaming platforms, to analyze audience sentiment, informing marketing strategies, content recommendations, or production decisions. The project achieves a test accuracy of 87.30% using a baseline LSTM model.
Dataset

Source: IMDB Dataset of 50,000 movie reviews, publicly available.
Attributes:
review: Textual movie reviews, including natural language with occasional HTML tags.
sentiment: Binary labels ("positive" or "negative").


Size: 50,000 reviews, balanced with ~25,000 positive and ~25,000 negative.
Preprocessing:
Label encoding: "positive" → 1, "negative" → 0.
Tokenization: Limited to 5,000 most frequent words.
Sequence padding: Standardized to 200 tokens.
Train-test split: 80% training (40,000 reviews), 20% testing (10,000 reviews).



Model

Architecture: LSTM-based deep learning model.
Embedding layer: 5,000-word vocabulary, 128-dimensional embeddings.
LSTM layer: 128 units, 0.2 dropout.
Dense layer: Sigmoid activation for binary classification.


Training:
Optimizer: Adam.
Loss: Binary Crossentropy.
Epochs: 5.
Batch Size: 64.
Validation Split: 0.2.


Variations Tested:
Baseline (128 units, 5 epochs, 0.2 dropout): 87.30% test accuracy.
Increased LSTM units (256 units): 86.95% test accuracy.
Increased epochs and reduced dropout (10 epochs, 0.1 dropout): 87.15% test accuracy.


Selected Model: Baseline model for its balance of accuracy and efficiency.

Repository Structure
├── Sentiment_Analysis_IMBD_Data_Google_colab.ipynb  # Jupyter Notebook with full code
├── Sentiment_Analysis_Report.pdf                    # Project report
├── model.h5                                        # Trained LSTM model
├── tokenizer.pkl                                   # Saved tokenizer
└── README.md                                       # This file

Setup Instructions
Prerequisites

Python 3.8+
Libraries: tensorflow, keras, pandas, numpy, scikit-learn, pickle
Optional: GPU for faster training (e.g., Google Colab with GPU runtime)

Installation

Clone the repository:git clone https://github.com/your-username/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb


Install dependencies:pip install tensorflow pandas numpy scikit-learn


Download the IMDB dataset (IMDB_Dataset.csv) and place it in the project directory. Available from Kaggle or other public sources.

Running the Code

Open Sentiment_Analysis_IMBD_Data_Google_colab.ipynb in Jupyter Notebook or Google Colab.
Ensure the dataset (IMDB_Dataset.csv) is in the correct path.
Run all cells to preprocess data, train the model, and evaluate performance.
Use the predictive_system function to classify new reviews, e.g.:print(predictive_system("This movie was amazing!"))



Results

Test Accuracy: 87.30% (baseline model).
Test Loss: 0.3397.
Key Findings:
The model accurately classifies positive and negative sentiments, suitable for stakeholder applications like content curation.
Balanced dataset ensures unbiased predictions.
Embedding layer captures semantic relationships effectively.


Limitations:
HTML tags in reviews may introduce noise.
Limited vocabulary (5,000 words) may miss rare sentiment-rich terms.
Slight overfitting observed in later epochs.



Next Steps

Clean HTML tags using BeautifulSoup.
Tune hyperparameters (e.g., vocabulary size, LSTM units).
Evaluate additional metrics (precision, recall, F1-score).
Explore transformer models like BERT for improved performance.
Augment with external data (e.g., Twitter reviews) for generalizability.

Usage Example
# Load model and tokenizer
from tensorflow.keras.models import load_model
import pickle
model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Predict sentiment
review = "A thrilling adventure with stunning visuals"
prediction = predictive_system(review)
print(f"Sentiment: {prediction}")  # Output: Sentiment: Positive

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

IMDB Dataset for providing a robust dataset for sentiment analysis.
TensorFlow and Keras for deep learning tools.
Google Colab for GPU support during development.

