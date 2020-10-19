
### Imports
import pandas as pd
import numpy as np

from gensim.models import Doc2Vec, LdaModel
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')

import pickle
import random
import streamlit as st

import pdfplumber

# import os
# import sys
# module_path = os.path.abspath(os.path.join('../..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
import utils.preprocess_helpers as ph



### functions
def load_df():
    ### load df
    CLEAN_DATA_PATH = './data/cleaned/'
    df_filename = 'monster_jobs_df_with_topics.pkl'
    with open(CLEAN_DATA_PATH+df_filename, 'rb') as f:
        df = pickle.load(f)
    return df

def load_topic_words():
    ### load df
    CLEAN_DATA_PATH = './data/cleaned/'
    topic_words_filename = '20_topics_100_words_each.pkl'
    with open(CLEAN_DATA_PATH+topic_words_filename, 'rb') as f:
        topic_words = pickle.load(f)
    return topic_words

def load_d2v_model():
    ### load d2v model for comparision
    MODEL_PATH = './models/'
    model_name = 'd2v_dm_0_vecsize_100_epochs_100.model'
    model = Doc2Vec.load(MODEL_PATH + model_name)
    return model

def load_LDA_model():
    MODEL_PATH = './models/'
    model_name = 'LDA_20_topics.model'
    model = LdaModel.load(MODEL_PATH+model_name)
    return model

def print_job_info(num):

    st.write('Job id: {}'.format(df.iloc[num]['id']))
    st.write('Job title: {}'.format(df.iloc[num]['job_title']))
    st.write('Company: {}'.format(df.iloc[num]['organization']))
    st.write('Job type: {}'.format(df.iloc[num]['job_type']))
    st.write('Sector: {}'.format(df.iloc[num]['sector']))
    st.write('Location: {}'.format(df.iloc[num]['location']))

    st.write('\nDescription: \n{}'.format(df.iloc[num]['job_description']) + '\n')

def predict_jobs(model, df, text, topn=20):
    """
    Predict similar jobs (held in dataframe) to new job (text input) using d2v model
    Converts string text into list, then uses infer_vector method to create infer_vector
    Uses cosine similarity for comparison
    """
    # print("\nSearching for matches for the following document: \n{}".format(text))
    pick_words = [word for word in simple_preprocess(str(text), deacc=True) if word not in STOPWORDS]
    pick_vec = model.infer_vector(pick_words, epochs=100, alpha=0.025)
    similars = model.docvecs.most_similar(positive=[pick_vec], topn=topn)

    print('\n'*4)
    print_similars(similars)

def print_similars(similars):
    st.write('\n'*4)
    count = 1
    for i in similars:
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('**Similar job number: {}**'.format(count))
        st.write('\n')
        count += 1
        st.write('Job ID ', i[0], ' Similarity Score: ', i[1])
        st.write('\n')

        print_job_info(i[0])

def get_doc_topics(document_string: str, model):
    doc_string = [document_string]
    doc_cleaned = ph.full_clean(doc_string)
    doc_dict = ph.make_dict(doc_cleaned)
    doc_bow = ph.make_bow(doc_cleaned, doc_dict)
    doc_lda = model[doc_bow]

    # only one document means only one element, a list of tuples with (topic, probability)
    for topic in doc_lda:
        # list of tuples
        doc_topics = topic
        # there should only be one list, so break
        break

    # sort in descending order
    return doc_topics


def show_skills_and_words(top_skills: int, skill_list: list):
    # skill_list is a a list of tuples
    skill_list = sorted(skill_list, reverse=True, key=lambda x: x[1])
    if top_skills > len(skill_list):
        top_skills = len(skill_list)
    for i in range(top_skills):
        skill = skill_list[i][0]
        score = skill_list[i][1]
        st.write('Skill #{}:'.format(i+1))
        st.write('Topic Grouping: {}  Score: {}'.format(skill, score))
        st.write('Skill words: {}'.format(topic_words[skill]))
        st.write(" ")
        st.write(" ")


def section_separator():
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("-"*50)
    st.write(" ")
    st.write(" ")
    st.write(" ")

def load_from_txt(selection, pdf=False):
    TXT_RAW_PATH = './data/raw/txt/'
    if pdf == True:
        TXT_RAW_PATH = './data/raw/pdf/txt_of_pdf/'
    with open(TXT_RAW_PATH + selection + '.txt', 'r') as f:
        # text_lookup_res = f.read().replace('\n', ' ')
        text_lookup_res = f.read()
    return text_lookup_res

def pdf_to_text(selection):
    PDF_RAW_PATH = './data/raw/pdf/'
    TO_TXT_PATH = './data/raw/pdf/txt_of_pdf/'
    temp = ''
    with pdfplumber.open(PDF_RAW_PATH + selection + '.pdf') as pdf:
        with open(TO_TXT_PATH + selection + '.txt', 'w') as write_to:
            for i in pdf.pages:
                temp = i.extract_text()
                write_to.write(temp + '\n')


##########################################################




"""
# Job Recommender Demo
Richard Kuzma, 1OCT2020
"""

"""
* Question 1: Given a resume, can we identify relevant skills?
* Question 2: Given a resume, can we recommend similar jobs?
* Question 3: Given a job, can we recommend similar jobs?
"""

file_option = st.selectbox('txt or pdf resume?', ['Select one', '.txt', '.pdf'])
#### text options
if file_option == 'Select one':
    st.warning('Please select either a .txt or .pdf example resume for the demo')
    st.stop()

elif file_option == '.txt':
    option = st.selectbox('which .txt resume would you like to use?',
                            ('Select one', 'Accounting', 'Data_Scientist', 'Logistics', 'Manufacturing_Engineer', 'Marketing', 'Nurse', 'Security_Guard', 'Software_Developer', 'Waitress'))
    if option == 'Select one':
        st.warning('Please select an example pdf resume for the demo')
        st.stop()
    text_lookup_res = load_from_txt(option.lower())

### pdf options
elif file_option == '.pdf':
    option = st.selectbox('which pdf resume would you like to use?',
                            ('Select one', 'Accountant', 'Auditor', 'Banking_Analyst', 'Business_Associate', 'Compliance', 'Investment_Banking', 'Investor_Relations', 'Office_Manager', 'Paralegal'))
    if option == 'Select one':
        st.warning('Please select an example pdf resume for the demo')
        st.stop()
    pdf_to_text(option.lower())
    text_lookup_res = load_from_txt(option.lower(), pdf=True)



# if option_txt == 'Select one' && option_pdf == 'Select one':
#     st.warning('Please select either a .txt or .pdf example resume for the demo')
#     st.stop()





st.write('## {} Resume Text:'.format(option))
st.write(text_lookup_res)

with st.spinner('Computing skills and job matches...'):
    df = load_df()
    d2v_model = load_d2v_model()
    lda_model = load_LDA_model()
    topic_words_all = load_topic_words()
st.success('Computation complete.')


section_separator()
st.write('## {} Resume Skills:'.format(option))
skill_words = 15
topic_words = [topic_words_all[i][:skill_words] for i in range(len(topic_words_all))]
with st.spinner('Extracting skills from resume...'):
    res_topics = get_doc_topics(text_lookup_res, lda_model)
    # st.write('Res topics ' + str(res_topics))
    # st.write('Ordered res topics ' + str(res_topics_ordered))

skills_to_display = st.slider('How many skills do you want to see?', 0, 20, 5)
show_skills_and_words(skills_to_display, res_topics)



    # top_skills = 4
    # if top_skills > len(res_topics_ordered):
    #     top_skills = len(res_topics_ordered)
    # for i in range(top_skills):
    #     skill = res_topics_ordered[i][0]
    #     score = res_topics_ordered[i][1]
    #     st.write('Skill #' + str(i+1) + ": " + str(skill) + ' score: ' + str(score))
    #     st.write('Skill words: ' + str(topic_words[skill]))

section_separator()
"""
## Jobs similar to this resume
"""
similar_jobs_to_resume = st.slider('# similar jobs to selected resume', 0, 15, 5)
predict_jobs(d2v_model, df, text=text_lookup_res, topn=similar_jobs_to_resume)


section_separator()
"""
## Search for Jobs Similar to a Selected Job
Pick a job number, see that job, you will be shown similar jobs
"""
job_num = int(st.text_input(label="Enter a Job ID between 0 and 22000", value="-1"), 10)
if job_num == -1:
    st.warning('No job ID selected for search')
    st.stop()

similar_jobs_to_job = st.slider('# similar jobs to selected job', 0, 10, 5)

st.write('#### Showing similar jobs to this one')
print_job_info(job_num)

# show similar jobs
text_lookup_job = df.iloc[job_num]['job_description']
predict_jobs(d2v_model, df, text=text_lookup_job, topn=similar_jobs_to_job)


section_separator()
"""
## Here's the data behind this demo
"""
short = df[:10000]
st.write(short)
