import streamlit as st
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from docx import Document
import PyPDF2

nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


def clean_resume(resume_text):
    cleanText = re.sub(
        r'http\S+|RT|cc|#\S+|@\S+|[%s]|[^\x00-\x7f]|[\s]+' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
        resume_text)
    return cleanText.strip()


def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        reader = PyPDF2.PdfFileReader(uploaded_file)
        text = ''
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
        return text
    elif uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    else:
        return uploaded_file.read().decode('utf-8')


def main():
    # Adding custom CSS
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f0f0;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-top: 20px;
        }
        .description {
            font-size: 24px;
            color: #555;
            text-align: center;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 18px;
        }
        .file-uploader {
            margin: 20px auto;
            display: flex;
            justify-content: center;
        }
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 200px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add a logo
    st.image('resume.png', use_column_width=True, output_format='PNG', caption="Resume Analyzer")

    st.markdown("<div class='title'>Resume Analyzer App</div>", unsafe_allow_html=True)
    st.markdown("<div class='description'>Upload your resume to find out the best-fit category.</div>",
                unsafe_allow_html=True)

    upload_file = st.file_uploader("Upload Resume", type=['txt', 'pdf', 'docx'], label_visibility="collapsed")

    if upload_file is not None:
        with st.spinner("Processing your resume..."):
            try:
                resume_text = extract_text_from_file(upload_file)
                if not resume_text:
                    st.error("Unable to extract text from the file.")
                    return
                cleaned_resume = clean_resume(resume_text)
                input_features = tfidf.transform([cleaned_resume])
                prediction_id = clf.predict(input_features)[0]

                # Map category ID to category name
                category_mapping = {
                    15: "Java Developer",
                    23: "Testing",
                    8: "DevOps Engineer",
                    20: "Python Developer",
                    24: "Web Designing",
                    12: "HR",
                    13: "Hadoop",
                    3: "Blockchain",
                    10: "ETL Developer",
                    18: "Operations Manager",
                    6: "Data Science",
                    22: "Sales",
                    16: "Mechanical Engineer",
                    1: "Arts",
                    7: "Database",
                    11: "Electrical Engineering",
                    14: "Health and fitness",
                    19: "PMO",
                    4: "Business Analyst",
                    9: "DotNet Developer",
                    2: "Automation Testing",
                    17: "Network Security Engineer",
                    21: "SAP Developer",
                    5: "Civil Engineer",
                    0: "Advocate",
                }
                category_name = category_mapping.get(prediction_id, "Unknown")
                st.success(f"Predicted category: {category_name}")

            except Exception as e:
                st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
