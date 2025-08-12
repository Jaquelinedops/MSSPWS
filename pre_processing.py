import os
import pandas as pd
import unicodedata
import re
import gc
import PyPDF2
import hashlib
import nltk
from pdfminer.high_level import extract_text
from sklearn.base import BaseEstimator, TransformerMixin
from wordsegment import load, segment



nltk.download('all')
import networkx as nx
from itertools import combinations
#Data analysis
import pandas as pd
import numpy as np
#Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set(font_scale=1)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer, confusion_matrix

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import CRF, scorers
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from nltk.corpus import wordnet
from collections import Counter
import ast
import scipy.stats
#import eli5
from textblob import TextBlob
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
import spacy

# Load spaCy's pre-trained model (English)
nlp = spacy.load("en_core_web_sm")

import os
from pdf2image import convert_from_path
import pytesseract



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import pytesseract

# Exemplo: se você instalou em C:\Program Files\Tesseract-OCR\tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# configurando pandas para não mostrar notação científica para números
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def hash_content(content):
    """
    Generates a SHA-256 hash for a given content.
    Ensures the input is treated as a string before hashing.
    """
    content_str = str(content)  # Convert to string to avoid errors
    return hashlib.sha256(content_str.encode()).hexdigest()


def clean_text(text):
    """ Cleans extracted text from PDFs by replacing/removing special and non-Latin characters """
    text = unicodedata.normalize("NFKD", text)  # Normalize Unicode

    # Replace common problematic characters
    #handle hiphenized words
    text = re.sub(r'-\s*\n\s*', '', text)
    text = text.replace(" -", "-")
    text = text.replace("-", "")
    text = text.replace("\n", " ").replace("\r", "").replace("\t", " ")
    text = text.replace("\u00AD", "").replace("\u00A0", " ").replace("\u200B", "")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2026", "...")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201C", "\"").replace("\u201D", "\"").replace("\ufeff", "")

    # Remove any other non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def ocr_pdf_to_df(pdf_path, dpi=300, poppler_path=None):
    filename = os.path.basename(pdf_path)
    try:
        pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    except Exception as e:
        print(f"Error converting PDF to images for {filename}: {e}")
        return pd.DataFrame()
    
    all_text = ""
    for i, page in enumerate(pages):
        print(f"Processing page {i+1} of {len(pages)} for file {filename}...")
        page_text = pytesseract.image_to_string(page)
        all_text += page_text + "\n"
    
    df = pd.DataFrame({"filename": [filename], "content": [all_text]})
    return df


def pdf_to_textsix(pdf_file_path):
    """Converts a single PDF file to cleaned text using pdfminer.six."""
    if os.path.exists(pdf_file_path):
        if os.path.getsize(pdf_file_path) == 0:  # Check if file is empty
            print(f"Skipping empty file: {pdf_file_path}")
            return None
        try:
            # Extract text from the PDF using pdfminer.six
            text = extract_text(pdf_file_path)
            if text:
                # Clean the extracted text with your custom clean_text function

                return clean_text(text)
            else:
                return None
        except Exception as e:
            print(f"Skipping corrupted PDF file: {pdf_file_path} (Error: {e})")
            return None
    else:
        print(f"File not found: {pdf_file_path}")
        return None

def pdfs_to_dataframe(directory_path):
    """
    Scans the given directory for PDF files, converts each to text using pdfminer.six,
    and returns a DataFrame with the results.
    
    :param directory_path: Path to the directory containing PDF files.
    :return: Pandas DataFrame with columns ['pdf_file', 'text']
    """
    data = []
    
    # Walk through the directory (non-recursive; use os.walk for recursive search if needed)
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            text = pdf_to_textsix(pdf_path)
            if text:
                data.append({"filename": filename, "content": text})
    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    return df

def count_unique_data(df):
  unique_count_dict = {}
  for col in df.columns:
    unique_count_dict[col] = df[col].unique().size
  return unique_count_dict

def split_camel_case_sentence(sentence):
    words = sentence.split()  # Split by spaces first
    corrected_words = [re.sub(r'([a-z])([A-Z])', r'\1 \2', word) for word in words]
    return " ".join(corrected_words)

def replace_terms(text, terms, replacement):
    # Use regex with case-insensitivity
    pattern = r'\b(' + '|'.join(map(re.escape, terms)) + r')\b'
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

def percentual_of_missing_data(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
    return missing_value_df

def compute_tfidf(text, top_n=20, lemmatize=False, encoding="iso-8859-1", n_gram_range=(2,3)):
    """
    Compute TF-IDF scores for the given text and return the top_n most relevant terms.

    :param text: The input text string.
    :param top_n: The number of top terms to return.
    :param lemmatize: Boolean flag to apply lemmatization.
    :param encoding: The text encoding to use.
    :param n_gram_range: The range of n-grams to be considered.
    :return: A DataFrame with terms and their TF-IDF scores.
    """
    # Ensure the text is not empty
    if not text.strip():
        return pd.DataFrame(columns=["Term", "TF-IDF Score"])

    if lemmatize:
        text = lemmatize_text(text)

    # Convert text into a list as required by TfidfVectorizer
    documents = [text]

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=top_n,
        encoding=encoding,
        ngram_range=n_gram_range
    )

    # Fit and transform the text
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Get feature names and corresponding TF-IDF scores from the document
    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()

    # Sort the terms by TF-IDF score in descending order
    sorted_indices = tfidf_scores.argsort()[::-1]
    top_features = feature_array[sorted_indices]
    top_scores = tfidf_scores[sorted_indices]

    # Build and return the resulting DataFrame
    result_df = pd.DataFrame({'Term': top_features, 'TF-IDF Score': top_scores})
    return result_df


def count_unique_terms(text):
    """
    Counts the number of unique words (terms) in a text.
    """
    # Normalize text: Convert to lowercase and remove non-alphabetic characters
    words = re.findall(r'\b\w+\b', text.lower())

    # Count occurrences of each word
    word_counts = Counter(words)

    # Number of unique words
    unique_word_count = len(word_counts)

    return unique_word_count, word_counts



# Example: Compute TF-IDF on the full dataset content

def get_only_words_from_strings(text):
    """
    Removes numbers, punctuation, and special characters from a given string.

    :param text: The input string to be cleaned.
    :return: A cleaned string with only alphabetic characters and spaces.
    """
    # Remove numbers, punctuation, and special characters
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    #remove digits
    cleaned_text = re.sub(r'\d+', ' ', cleaned_text ).strip()
    return cleaned_text



# Function to map POS tags from NLTK to WordNet
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()  # Get first letter of POS tag
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN

# Function to lemmatize a sentence with correct POS tagging
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)  # Tokenize text

    lemmatized_words = [lemmatizer.lemmatize(word, pos='n') for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return " ".join(lemmatized_words)


def remove_stop_words(text, custom_stop_words = ''):
    """
    Remove common English stop words from the input text.
    
    :param text: Input string.
    :return: A string with stop words removed.
    """
    stop_words = set(stopwords.words('english'))
    
    # Process personalized stop words if provided
    if custom_stop_words:
        # If personalized_stop_words is a string, split it into words
        if isinstance(custom_stop_words, str):
            extra_words = set(custom_stop_words.split())
        # If it's already a list or set, just convert it to a set
        elif isinstance(custom_stop_words, (list, set)):
            extra_words = set(custom_stop_words)
        else:
            extra_words = set()
        # Merge the extra words with the default stop words
        stop_words = stop_words.union(extra_words)
    
    # Tokenize the text (simple whitespace split)
    words = text.split()
    # Filter out stop words (case-insensitive check)
    filtered_words = [word for word in words if word.lower().strip() not in stop_words]
    # Rejoin the filtered words into a single string
    return " ".join(filtered_words)



def extract_short_words(text):
    """
    Extracts and returns a list of words with 3 or fewer letters from the input text.
    
    :param text: Input string.
    :return: List of words (strings) with 3 or fewer letters.
    """
    # This regex pattern matches word boundaries around 1 to 3 alphabetical characters.
    # Adjust the pattern if you want to include numbers or other characters.
    short_words = re.findall(r'\b[a-zA-Z]{1,3}\b', text)
    return short_words

def identify_frequent_short_words(text, top_n=100, max_length=3):
    """
    Identifies the top_n most frequent words in the input text that have max_length or fewer letters.
    
    :param text: Input string.
    :param top_n: Number of top frequent words to consider.
    :param max_length: Maximum length of words to include.
    :return: A list of tuples (word, count) for words with max_length or fewer letters.
    """
    # Use regex to extract words and convert them to lowercase
    words = re.findall(r'\w+', text.lower())
    # Count the frequency of each word
    word_counts = Counter(words)
    # Get the top_n most common words
    most_common = word_counts.most_common(top_n)
    # Filter out words with more than max_length letters
    filtered = [ (word, count) for word, count in most_common if len(word) <= max_length ]
    return filtered



def correct_text(text):
    """
    Corrects spelling errors in a given text using TextBlob.
    
    :param text: A string containing text.
    :return: The corrected text.
    """
    if isinstance(text, str):  # Ensure input is a string
        return str(TextBlob(text).correct())
    return text  # Return as-is if not a string


def plot_filenames_for_entity(entity_value, df, entity_col='entity', filename_col='name'):
    """
    Plots a graph linking a chosen entity (e.g., a location) to all filenames that contain that entity,
    using a shell layout for better visualization.

    The entity node is placed in the center and all associated filenames are arranged in an outer ring.

    :param entity_value: The value of the entity to filter (e.g., "new york")
    :param df: A DataFrame containing at least the filename and entity column.
    :param entity_col: Name of the column that contains the entity (default: 'entity')
    :param filename_col: Name of the column that contains filenames (default: 'name')
    """
    # Filter the DataFrame to only include rows with the chosen entity
    subset = df[df[entity_col].str.lower() == entity_value.lower()]
    if subset.empty:
        print(f"No filenames found for {entity_col} '{entity_value}'")
        return

    # Drop duplicate file names to avoid redundant edges
    file_nodes = subset[filename_col].dropna().unique()

    # Create a graph
    G = nx.Graph()

    # Add the central entity node
    G.add_node(entity_value, node_type='entity')

    # Add file nodes and connect each to the entity
    for filename in file_nodes:
        G.add_node(filename, node_type='file')
        G.add_edge(entity_value, filename)

    # Use a shell layout with two shells: one for the central entity, one for the filenames.
    shells = [[entity_value], list(file_nodes)]
    pos = nx.shell_layout(G, nlist=shells)

    # Define colors: central entity gets one color; files get another.
    node_colors = ["lightgreen" if node == entity_value else "skyblue" for node in G.nodes()]

    plt.figure(figsize=(16, 12))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1200,
        font_size=10,
        edge_color='gray'
    )
    
    plt.title(f"'{entity_col}' related to '{entity_value}'", fontsize=16)
    plt.axis('off')
    plt.show()



def extract_unique_values(df, column):
    """
    Extract unique values from a column containing stringified lists.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name containing the lists.

    Returns:
    list: A list of unique values extracted from the column.
    """
    # Convert string lists to actual Python lists
    df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Flatten the list and extract unique values
    unique_values = list(set(item for sublist in df[column] for item in sublist))

    return unique_values

# Example usage
# df = pd.DataFrame({'person': ["['Alice', 'Bob']", "['Bob', 'Charlie']", "['Alice', 'David']"]})
# unique_persons = extract_unique_values(df, 'person')
# print(unique_persons)


# Example usage:
# df = replace_entities_with_placeholder(df, 'only_words_content', 'person', '[PERSON]')
# df = replace_entities_with_placeholder(df, 'only_words_content', 'location', '[LOCATION]')
# df = replace_entities_with_placeholder(df, 'only_words_content', 'org', '[ORG]')

def run_ocr_on_files(pdf_folder, list_ocr_files, poppler_path, dpi=300):
    """
    Loops through each PDF file in the list, runs OCR using ocr_pdf_to_df,
    and returns a concatenated DataFrame with OCR results.
    
    :param pdf_folder: Folder containing the PDF files.
    :param list_ocr_files: List of PDF filenames.
    :param poppler_path: Path to the poppler 'bin' directory.
    :param dpi: Resolution for converting PDF pages to images (default 300).
    :return: A DataFrame with columns 'filename' and 'content'.
    """
    dfs = []  # List to hold dataframes for each PDF
    for filename in list_ocr_files:
        pdf_path = os.path.join(pdf_folder, filename)
        
        # Check if the file exists before processing
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            continue
        
        # Perform OCR on the PDF using the previously defined function
        temp_df = ocr_pdf_to_df(pdf_path, dpi=dpi, poppler_path=poppler_path)
        dfs.append(temp_df)
    
    # Concatenate all the individual DataFrames into one
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        return final_df
    else:
        print("No data frames were created. Check for errors or empty list.")
        return pd.DataFrame()

# Example usage:
pdf_folder = r"C:\Users\Jaque\Downloads\List_of_cases\List_of_cases"

poppler_path = r"C:\Users\Jaque\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

#final_df = run_ocr_on_files(pdf_folder, list_ocr_files, poppler_path)
#print(final_df.head())
#df = pd.concat([final_df, df], ignore_index=True)
#df["content_length"] = df["content"].str.len()
#df.sort_values("content_length", ascending=False, inplace=True)
#df.drop_duplicates(subset="filename", keep="first", inplace=True)
#df.reset_index(inplace=True)
#df.to_csv("modern_slavery_cases2.csv")
# Load word database

def find_emails(text):
    # Regular expression pattern for emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    
    # Find all matches in the text
    emails = re.findall(email_pattern, text)
    
    return emails

def find_links(text):
    # Regex to match valid URLs (http(s), www, or plain domain with TLD)
    pattern = r'\b(?:https?://|www\.)[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:/[^\s]*)?|\b[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?\b'
    
    # Get all matches
    matches = re.findall(pattern, text)
    
    # Optional: remove duplicates and sort
    return list(sorted(set(matches)))

def plot_term_frequency(tfidf_df, top_n=50):
    """
    Plots a bar graph for term frequency using TF-IDF scores.

    :param tfidf_df: DataFrame containing terms and their TF-IDF scores.
    :param top_n: Number of top terms to display.
    """
    # Select top N terms

    top_terms = tfidf_df.head(top_n)

    # Plot
    plt.figure(figsize=(12, 12))
    plt.barh(top_terms["Term"], top_terms["TF-IDF Score"], color="skyblue")
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Terms")
    plt.title("Top Term Frequencies Based on TF-IDF")
    plt.gca().invert_yaxis()  # Highest scores on top
    plt.show()

def is_undesired_entity(term):
    """
    Returns True if the term contains any entity matching the undesired labels.
    """
    doc = nlp(term)
    for ent in doc.ents:
        if ent.label_:
            return True
    return False

# Assuming tfidf_results is your DataFrame with columns 'Term' and 'Aggregated TF-IDF Score'
# Filter out rows whose term is recognized as an undesired entity
#filtered_df = tfidf_results[~tfidf_results['Term'].apply(is_undesired_entity)]


def extract_entities(text, labelset=("LOC")):
    """
    Extracts unique entities (of the given labelset) from the input text.
    If the input is not a string, it returns an empty list.
    
    :param text: Input text (expected to be a string).
    :param labelset: Tuple of entity labels to extract.
    :return: A list of unique entity texts.
    """
    # Check if the text is a string; if not, return empty list
    if not isinstance(text, str):
        return []
    
    max_chunk = 1000000
    all_entities = []
    
    # Split the text into chunks if it exceeds max_chunk
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    
    for chunk in chunks:
        doc = nlp(chunk)
        # Filter entities by labelset
        chunk_ents = [ent.text for ent in doc.ents if ent.label_ in labelset]
        all_entities.extend(chunk_ents)
    
    # Remove duplicates while preserving order (case insensitive)
    seen = set()
    unique_entities = []
    for e in all_entities:
        e_lower = e.lower().strip()
        if e_lower not in seen:
            seen.add(e_lower)
            unique_entities.append(e)
    
    return unique_entities


def replace_entity(text, entities, placeholder):
    if not isinstance(text, str) or not isinstance(entities, list):  # Ensure valid inputs
        return text

    for entity in entities:
        if entity:  # Avoid empty strings
            pattern = r'\b' + re.escape(entity) + r'\b'  # Match whole words only
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)  # Ensure case insensitivity
    
    return text

# Define a custom transformer for lemmatization
class LemmatizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [" ".join([self.lemmatizer.lemmatize(word) for word in word_tokenize(text)])
                if isinstance(text, str) else text for text in X]



def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'], title='Confusion Matrix with Percentages', cmap='Blues'):
    """
    Plots a confusion matrix with both absolute values and percentages.

    Parameters:
    - y_true: array-like, true class labels.
    - y_pred: array-like, predicted class labels.
    - labels: list, class labels for x and y axes.
    - title: str, title of the plot.
    - cmap: str, color map for the heatmap.
    """
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Convert to percentages
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100

    # Create annotations combining absolute and percentage values
    annot_labels = np.array([
        [f"{conf_matrix[i, j]}\n({conf_matrix_percent[i, j]:.1f}%)"
         for j in range(conf_matrix.shape[1])]
        for i in range(conf_matrix.shape[0])
    ])

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(conf_matrix, annot=annot_labels, fmt='', cmap=cmap, square=True, linewidths=0.5, linecolor='black')

    # Labels and formatting
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xticklabels(labels, fontsize=10)  # Adjust labels if needed
    ax.set_yticklabels(labels, fontsize=10)
    plt.title(title, fontsize=14)

    # Show the plot
    plt.show()


def remove_numbers(text):
    """
    Removes numbers from a sentence, replaces them with a single space,
    and removes all double spaces.

    :param text: A string containing text.
    :return: The cleaned text.
    """
    if not isinstance(text, str):
        return text  # Return as-is if not a string
    
    text = re.sub(r'\d+', ' ', text)  # Replace numbers with a single space
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


from urllib.parse import urlparse

# Extracting the radical domain name
def extract_radical(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc  # Extracts domain part
    domain = domain.replace("www.", "")  # Removes "www." if present
    return domain



# Example
from wordsegment import load, segment
load()
def detect_words_segment(text):
    return " ".join(segment(text))


