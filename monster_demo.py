
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

def choose_res_text(option):
    text_lookup_res = ""
    if option == 'Nurse':
        example_registered_nurse1 = "Certified Registered Nurse Anesthetist Job Responsibilities:  Provides pre-anesthetic preparation and patient evaluation. Ensures patient identification and obtains appropriate health history. Recommends, requests, and evaluates pertinent diagnostic studies. Documents pre-anesthetic evaluation. Obtains informed consent for anesthesia. Selects and/or administers pre-anesthetic medications. Selects prepares and administers anesthetic agents or other agents administered in management of anesthetic care. Informs anesthesiologist and/or surgeon of changes in patient’s condition. Provides anesthesia induction, maintenance, emergence, and post anesthesia care. Inserts invasive line catheter/devices. Performs tracheal intubation and extubation, airway management. Provides mechanical ventilation. Performs venous and arterial punctures. Obtains blood samples. Performs and manages regional anesthetic. Manages patient’s fluid, blood, electrolyte and acid base"
        example_registered_nurse2 = "balance. Provides perianesthetic invasive and non-invasive monitoring utilizing current standards and techniques. Responds to abnormal findings with corrective action. Recognizes and treats cardiac dysrhythmias through use of perianesthetic electrocardiogram monitoring. Evaluates patient response during emergence from anesthesia. Institutes pharmacological or supportive treatment to insure adequacy of patient recovery from anesthesia and adjuvant drugs. Provides post anesthesia follow-up, report, and evaluation of patient’s response to anesthesia and for potential anesthetic complication. Identifies and manages emergency situations. Initiates or participates in cardiopulmonary resuscitation. Performs or orders equipment safety checks as needed. Cleans and sterilizes equipment and notifies supervisor of needed equipment adjustments/repairs. May perform patient care to the extent necessary to maintain clinical expertise, competency and licensing"
        example_registered_nurse3 = "necessary to fulfill job responsibilities and to direct the provision of care on the unit. Education, Experience, and Licensing Requirements:  Graduate of accredited nurse anesthesia program To (2) years of anesthesia care in acute setting experience preferred Valid state RN License (must meet education requirement(s) for state licensure) Valid state APRN Recognition (must meet education requirement (s) for state recognition and obtain within six (6) months of hire) Certified Registered Nurse Anesthetist (CRNA) by the American Association of Nurse Anesthetists (AANA) Certified Advanced Cardiac Life Support (ACLS) by the American Heart Association Current BLS for Healthcare Provider CPR or CPR/AED for the Professional Rescuer certification National Provider Identifier (NPI) and Taxonomy code required at time of hire"
        text_lookup_res = example_registered_nurse1 + ' ' + example_registered_nurse2 + ' ' + example_registered_nurse3
    elif option == 'Data Engineer':
        example_data_engineer1 = "Data Engineer Job Responsibilities:  Develops and maintains scalable data pipelines and builds out new API integrations to support continuing increases in data volume and complexity. Collaborates with analytics and business teams to improve data models that feed business intelligence tools, increasing data accessibility and fostering data-driven decision making across the organization. Implements processes and systems to monitor data quality, ensuring production data is always accurate and available for key stakeholders and business processes that depend on it. Writes unit/integration tests, contributes to engineering wiki, and documents work. Performs data analysis required to troubleshoot data related issues and assist in the resolution of data issues. Works closely with a team of frontend and backend engineers, product managers, and analysts. Defines company data assets (data models), spark, sparkSQL, and hiveSQL jobs to populate data models. Designs"
        example_data_engineer2 = "data integrations and data quality framework. Designs and evaluates open source and vendor tools for data lineage. Works closely with all business units and engineering teams to develop strategy for long term data platform architecture.  Data Engineer Qualifications / Skills:  Knowledge of best practices and IT operations in an always-up, always-available service Experience with or knowledge of Agile Software Development methodologies Excellent problem solving and troubleshooting skills Process oriented with great documentation skills Excellent oral and written communication skills with a keen sense of customer service Education, Experience, and Licensing Requirements:  BS or MS degree in Computer Science or a related technical field 4+ years of Python or Java development experience 4+ years of SQL experience (No-SQL experience is a plus) 4+ years of experience with schema design and dimensional data modeling Ability in managing and communicating data"
        example_data_engineer3 = "warehouse plans to internal clients E perience designing, building, and maintaining data processing systems Experience working with either a Map Reduce or an MPP system on any size/scale"
        text_lookup_res = example_data_engineer1 + ' ' + example_data_engineer2 + ' ' + example_data_engineer3
    elif option == 'Business Analyst':
        example_biz_analyst1 = "Business Analyst Job Responsibilities:  Elicits, analyzes, specifies, and validates the business needs of stakeholders, be they customers or end users. Collaborates with project sponsors to determine project scope and vision. Clearly identifies project stakeholders and establish customer classes, as well as their characteristics. Conducts interviews to gather customer requirements via workshops, questionnaires, surveys, site visits, workflow storyboards, use cases, scenarios, and other methods. Identifies and establishes scope and parameters of requirements analysis on a project-by-project basis to define project impact, outcome criteria, and metrics. Works with stakeholders and project team to prioritize collected requirements. Researches, reviews, and analyzes the effectiveness and efficiency of existing requirements-gathering processes and develop strategies for enhancing or further leveraging these processes. Assists in conducting research on products"
        example_biz_analyst2 = "to meet agreed upon requirements and to support purchasing efforts. Participates in the QA of purchased solutions to ensure features and functions have been enabled and optimized. Participates in the selection of any requirements documentation software solutions that the organization may opt to use. Analyzes and verifies requirements for completeness, consistency, comprehensibility, feasibility, and conformity to standards. Develops and utilizes standard templates to accurately and concisely write requirements specifications. Translates conceptual customer requirements into functional requirements in a clear manner that is comprehensible to developers/project team. Creates process models, specifications, diagrams, and charts to provide direction to developers and/or the project team. Develops and conduct peer reviews of the business requirements to ensure that requirement specifications are correctly interpreted. Assists with the interpretation of customer"
        example_biz_analyst3 = "requirements into feasible options, and communicating these back to the business stakeholders. Manages and tracks the status of requirements throughout the project lifecycle; enforce and redefine as necessary. Communicates changes, enhancements, and modifications of business requirements — verbally or through written documentation — to project managers, sponsors, and other stakeholders so that issues and solutions are understood. Provides guidance and/or instruction to junior staff members."
        text_lookup_res = example_biz_analyst1 + ' ' + example_biz_analyst2 + ' ' + example_biz_analyst3
    # if st.checkbox('Enter resume text manually'):
    #     text_lookup_res = st.text_input(label="Enter resume text")
    return text_lookup_res

def section_separator():
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("-"*50)
    st.write(" ")
    st.write(" ")
    st.write(" ")

##########################################################




"""
# Job Recommender Demo
Richard Kuzma, 1OCT2020
"""
"""
* Question 1: Given a resume, can we recommend similar jobs?
* Question 2: Given a job, can we find similar jobs?
"""


option = st.selectbox('which resume would you like to use?',
                        ('Select one', 'Nurse', 'Data Engineer', 'Business Analyst'))
if option == 'Select one':
    st.warning('Please select an example resume for the demo')
    st.stop()

text_lookup_res = choose_res_text(option)

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
