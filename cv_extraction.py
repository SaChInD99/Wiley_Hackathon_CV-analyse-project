import streamlit as st
import fitz
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tempfile
import pymongo

nltk.download(['stopwords', 'wordnet']) 



# Set the background color to light gray
st.markdown(
    """
    <style>
    body {
        background-color: linear-gradient(black, grey);
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_resources():
    nlp = spacy.load('your_trained_model')
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.from_disk("Technical_skills.jsonl")
    return nlp

nlp = load_resources()

@st.cache_data
def extract_text(file):
    tempfile.mkstemp() 
    with open("temp_file.pdf", "wb") as f:
        f.write(file.read()) 

    doc = fitz.open("temp_file.pdf")
    # doc = fitz.open(file)
    text = []
    for page in doc:
        text.append(page.get_text())
    return " ".join(text)  

def preprocess(text):
    text = text.lower() 
    text = text.split()  
    lm = WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text = " ".join(text)
    return text

# Connect to MongoDB
#client = pymongo.MongoClient("mongodb+srv://nawodyaa59:zLnU8ZSjBaBqSyq5@cluster0.7c9abpa.mongodb.net/?retryWrites=true&w=majority")
client = pymongo.MongoClient("mongodb+srv://dimuthcbandara97:5DmqaiwsDk0WFp2M@cvanalysis.tata8.mongodb.net/?retryWrites=true&w=majority")
db = client["CVANALYSE"]
collection = db["cvanalyse"]

st.title("CV Analysis App")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

with st.form("cv_form"):
    st.write("Please fill out the following fields:")
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email Address")
    phone_number = st.text_input("Phone Number")
    cv_file = st.file_uploader("Upload Resume", type=["pdf"])

    submitted = st.form_submit_button("Submit")

if submitted:
    cv_text = extract_text(cv_file)
    cleaned_text = preprocess(cv_text)
    
    doc = nlp(cleaned_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    st.header("Extracted Entities")
    labels = ["PERSON", "JOB", "CONTACT", "ORG", "EDUCATION", "LOCATION", "LINK", "TECHNICAL SKILLS", "NON TECHNICAL SKILLS"]
    num_cols = len(labels)
    cols = st.columns(num_cols)
    for i, label in enumerate(labels):
        col_entities = [entity[0] for entity in entities if entity[1] == label]
        cols[i].write(f"{label}")
        cols[i].write(col_entities)

        # Save form data and processed CV data to MongoDB
    cv_data = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone_number": phone_number,
        "cv_text": cleaned_text,
        "entities": entities
        # "cv_file": cv_file.read()
    }
    
    collection.insert_one(cv_data)

    st.write("Form data and processed CV data saved to MongoDB!")