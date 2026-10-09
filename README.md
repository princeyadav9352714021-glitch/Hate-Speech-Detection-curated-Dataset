# Hate Speech Detection using NLP
---
<img src="demo.gif" width="500" alt="Alt Text" />


---
## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning
to classify text into two categories:

-   **Label 0:** Not Hate Speech
-   **Label 1:** Hate Speech / Offensive Content

The notebook downloads the dataset through KaggleHub, explores and
cleans the data, converts text into TF-IDF features, trains a Logistic
Regression model, evaluates its performance, saves the model and
vectorizer, and demonstrates a prediction on user-entered text.

> **Note:** This is a text-classification project for learning and
> demonstration. Predictions may be wrong, and the model may miss
> context, sarcasm, slang, or the intended meaning of a sentence. Do not
> use its output as the sole basis for decisions about people.

## Technologies Used

-   Python
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   NLTK
-   Scikit-learn
-   KaggleHub
-   Joblib
-   Jupyter Notebook

## Dataset

The notebook downloads the dataset using KaggleHub:

`waalbannyantudre/hate-speech-detection-curated-dataset`

It then loads `HateSpeechDatasetBalanced.csv` and works with the
`Content` and `Label` columns.

The dataset file must be present in the downloaded folder. If KaggleHub
cannot download it, check your internet connection and Kaggle
access/setup.

## Project Workflow

1.  Import the required Python libraries.
2.  Download the dataset using KaggleHub.
3.  Load the CSV file and inspect its shape, columns, data types, and
    summary statistics.
4.  Check missing values and visualize them.
5.  Check for duplicate rows and remove duplicates.
6.  Inspect the distribution of the target `Label` and plot the label
    counts.
7.  Download NLTK English stopwords.
8.  Clean text by converting it to lowercase, removing URLs and
    non-letter characters, and removing English stopwords.
9.  Create `processed_text` and examine text length.
10. Convert cleaned text into TF-IDF features with up to 20,000 features
    and unigram/bigram combinations.
11. Split the data into training and testing sets (80% training, 20%
    testing), using stratification.
12. Train a Logistic Regression classifier.
13. Print training and testing accuracy.
14. Save and reload the model and TF-IDF vectorizer.
15. Enter a sentence and generate a prediction.

## Installation

Use Python 3.10 or a compatible Python version, then install the
dependencies in your VS Code terminal:

``` bash
python -m pip install pandas numpy matplotlib seaborn nltk scikit-learn kagglehub joblib jupyter
```

If you plan to run a separate Streamlit frontend, install Streamlit too:

``` bash
python -m pip install streamlit
```

## Run the Notebook

1.  Download or clone this project folder.
2.  Open the folder in VS Code.
3.  Install the dependencies listed above.
4.  Open `Hate_Speech_Detection_curated_Dataset.ipynb`.
5.  Select a Python kernel.
6.  Run the notebook cells from top to bottom.

The first run requires an internet connection for KaggleHub to download
the dataset and NLTK to download stopwords.

## Saved Model Files

The notebook saves these files in the current working directory:

-   `hate_speech_model.pkl` --- trained Logistic Regression model
-   `tfidf_vectorizer.pkl` --- fitted TF-IDF vectorizer

Keep both files from the same training run. The vectorizer used by the
application must be the same fitted vectorizer used to train the model.

## Example Prediction

The final notebook cell accepts text from the console, cleans it,
transforms it with the saved TF-IDF vectorizer, and predicts the class.

-   Prediction `1`: Hate Speech / Offensive Content
-   Prediction `0`: Not Hate Speech

These labels describe the dataset's classes; they are not guaranteed
judgments about the meaning or intent of a text.

## Important Preprocessing Note

Any frontend that uses the saved model must apply the same
`clean_text()` preprocessing used during training. This notebook removes
URLs, non-letter characters, and English stopwords. If the frontend
preprocesses text differently, model predictions may change.

## Limitations

-   The model can make false-positive and false-negative predictions.
-   Short text may not provide enough context.
-   Removing stopwords and punctuation can remove information useful for
    interpreting meaning.
-   The model's probability estimates, if used, should not be treated as
    verified truth.
-   Accuracy alone does not show how well the model handles each class;
    review the confusion matrix, precision, recall, and F1-score when
    evaluating it further.

## Future Improvements

-   Evaluate the model with a confusion matrix and classification
    report.
-   Test on text that was not used during training.
-   Compare models using the same train/test split.
-   Build a Streamlit interface for interactive predictions.
-   Monitor false positives and false negatives and document dataset
    limitations.

## Project Structure

``` text
Hate-Speech-Detection/
├── Hate_Speech_Detection_curated_Dataset.ipynb
├── hate_speech_model.pkl
├── tfidf_vectorizer.pkl
├── app.py                 # Optional Streamlit frontend
└── README.md
```

The `.pkl` files are created after running the model-saving notebook
cell. `app.py` is optional and is not generated by the notebook itself.
