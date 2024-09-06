from flask import Blueprint, redirect, render_template, request, flash, jsonify, url_for
from flask_login import login_required, current_user
from sqlalchemy import or_
from .models import Drugs, Info
from . import db
import json
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import requests
from pytrials.client import ClinicalTrials
import re
from fuzzywuzzy import process
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
import time

views = Blueprint('views', __name__)

@views.route('/')
def opening():
    logged_in = False
    if current_user.is_authenticated:
        logged_in = True
    
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template('opening-mobile.html', logged_in=logged_in, user=current_user)
    else:
        return render_template('opening.html', logged_in=logged_in, user=current_user)

#start of helper functions for model
def get_study_ids(drug, disease, df):
    # Filter rows based on drug
    # drug_mask = df["Interventions"].str.contains(drug, case=False, na=False)
    # # Filter rows based on disease
    # disease_mask = df["Conditions"].str.contains(disease, case=False, na=False)
    drug_regex = re.compile(re.escape(drug.strip()), re.IGNORECASE)
    disease_regex = re.compile(re.escape(disease.strip()), re.IGNORECASE)

    # Filter rows based on drug using regex
    drug_mask = df["Interventions"].apply(lambda x: bool(drug_regex.search(str(x))) if pd.notna(x) else False)
    # Filter rows based on disease using regex
    disease_mask = df["Conditions"].apply(lambda x: bool(disease_regex.search(str(x))) if pd.notna(x) else False)
    # Combine both masks
    combined_mask = drug_mask & disease_mask
    # Get study IDs for the filtered rows
    return df[combined_mask]['NCT Number'].tolist()

def fetch_data_with_retries(url, max_retries=3, backoff_factor=0.3):
    for retry in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except (ChunkedEncodingError, ConnectionError, Timeout) as e:
            if retry < max_retries - 1:
                time.sleep(backoff_factor * (2 ** retry))  # Exponential backoff
            else:
                raise
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

def sort(studies_list):
    # studies_list = ast.literal_eval(studies_list)
    # Query the database directly where 'Study NCT' is in studies_list
    query_result = Info.query.filter(Info.NCT.in_(studies_list)).all()
    # Convert query result to a list of dictionaries (if not already in this format)
    df_sorted = [{'Study NCT': item.NCT, 'Number of participants': item.num_participants,
                  'Number of female participants': item.num_women,
                  'Number of male participants': item.num_men} for item in query_result]
    
    # Sort the list of dictionaries
    df_sorted.sort(key=lambda x: x['Number of participants'], reverse=True)
    
    # Extract top 3 studies
    top_3 = [item['Study NCT'] for item in df_sorted[:3]]
    
    # Calculate sums and proportions
    sum_females = sum(item['Number of female participants'] for item in df_sorted[:3])
    sum_males = sum(item['Number of male participants'] for item in df_sorted[:3])
    total_participants = sum(item['Number of participants'] for item in df_sorted[:3])
    
    if total_participants > 0:
        female_proportion = sum_females / total_participants * 100
        male_proportion = sum_males / total_participants * 100
    else:
        female_proportion = -1.00
        male_proportion = -1.00

    return (top_3, total_participants, sum_females, sum_males, female_proportion, male_proportion)

# def divide_by_100(x):
#     return x/100

def get_study(studies_related_pair_input):
    tot_num_females = 0
    tot_num_males = 0
    total_num_participants = 0
    female_proportion = 0
    male_proportion = 0

    num_studies = len(studies_related_pair_input)
    
    #tbd
    ct = ClinicalTrials()
    api_url = 'https://clinicaltrials.gov/api/v2/studies/'

    info1 = Info.query.all()
    info_dict = {info.NCT: {'num_women': info.num_women, 'num_men': info.num_men} for info in info1}

    for study_id in studies_related_pair_input:
        try:
            female_count = 0 
            male_count = 0
            if study_id in info_dict:
                female_count = info_dict[study_id]['num_women']
                male_count = info_dict[study_id]['num_men']
            else:
                data = fetch_data_with_retries(api_url + study_id)

                if data.get("hasResults", False):
                    # Initialize counters for female and male participants
                    # study_path = f"{study_id}.json"
                    # with open(study_path, 'w') as json_file:
                    #     json.dump(data, json_file)
                    female_count = 0
                    male_count = 0

                # Parse the JSON data to count the participants
                if "resultsSection" in data and "baselineCharacteristicsModule" in data["resultsSection"]:
                    for measure in data["resultsSection"]["baselineCharacteristicsModule"]["measures"]:
                        if measure["title"] == "Sex: Female, Male":
                            for category in measure["classes"][0]["categories"]:
                                if category["title"] == "Female":
                                    female_values = []
                                    if "measurements" in category:
                                        for measurement in category["measurements"]:
                                            if measurement["value"].replace('.', '', 1).isdigit():
                                                female_values.append(int(float(measurement["value"])))
                                    female_count = sum(female_values)
                                elif category["title"] == "Male":
                                    male_values = []
                                    if "measurements" in category:
                                        for measurement in category["measurements"]:
                                            if measurement["value"].replace('.', '', 1).isdigit():
                                                male_values.append(int(float(measurement["value"])))
                                    male_count = sum(male_values)
                    
                    info = Info(NCT=study_id, num_men = male_count, num_women = female_count, num_participants=male_count+female_count)
            tot_num_females += female_count
            tot_num_males += male_count
        except Exception as e:
            print(f"Failed to fetch data for study ID {study_id}: {e}")

    total_num_participants = tot_num_females + tot_num_males

    if total_num_participants > 0:
        # Calculate proportions
        female_proportion = round((tot_num_females / total_num_participants) * 100, 2)
        male_proportion = round((tot_num_males / total_num_participants) * 100, 2)
    else:
        female_proportion = -1.00
        male_proportion = -1.00

    return num_studies, female_proportion, male_proportion, tot_num_females, tot_num_males

def get_model(drug, disease):
    # model_file_path = r'C:\Users\anbgo\coding_projects_flask\GitHub-Rx\project_new\trial_something\regression_model.pkl'
    # preprocessor_file_path =  r'C:\Users\anbgo\coding_projects_flask\GitHub-Rx\project_new\trial_something\preprocessor.pkl'

    model_file_path = os.path.join(os.path.dirname(__file__), 'regression_model.pkl')
    preprocessor_file_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')
    
    BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    csv_file1 = os.path.join(BASE_DIR1, 'ctg-studies.csv')  # Join the base directory with the file name
    df1 = pd.read_csv(csv_file1)  

    #cleaning names
    df1["Conditions"] = df1["Conditions"].str.upper()
    df1["Interventions"] = df1["Interventions"].str.upper()
    df1["Conditions"] = df1["Conditions"].fillna('').astype(str)
    df1["Interventions"] = df1["Interventions"].fillna('').astype(str)

 
    # Load processor
    with open(preprocessor_file_path, 'rb') as f:
        preprocessor = pickle.load(f)

    # Load model
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)

    nct_list = get_study_ids(drug, disease, df1)
    num_studies, female_proportion, male_proportion, tot_num_females, tot_num_males = get_study(nct_list)

    (top_3, total_participants, sum_females, sum_males, female_proportion1, male_proportion1) = sort(nct_list)

    sample_data = pd.DataFrame({
        'Indication': [disease],
        'Num Studies': [num_studies],
        'Total females in studies': [tot_num_females],
        'Total males in studies': [tot_num_males],
        'Male proportion in studies': [male_proportion],
        'Female proportion in studies': [female_proportion],
        'Number of participants in most relevant studies': [total_participants],
        'Number of female participants in most relevant studies': [sum_females],
        'Number of male participants in most relevant studies': [sum_males],
        'Proportion of females in most relevant studies': [female_proportion1],
        'Proportion of males in most relevant studies': [male_proportion1]
    })

    transformed_data = preprocessor.transform(sample_data)

    transformed_columns = (
        preprocessor.transformers_[0][1].get_feature_names_out(['Indication']).tolist() +
        [
            # 'Indication',
            # 'Num Studies',
            # 'Total females in studies',
            # 'Total males in studies',
            # 'Male proportion in studies',
            # 'Female proportion in studies',
            # 'Number of participants in most relevant studies',
            # 'Number of female participants in most relevant studies',
            # 'Number of male participants in most relevant studies',
            # 'Proportion of females in most relevant studies',
            # 'Proportion of males in most relevant studies'
            'Total females in studies', 
            'Total males in studies',
            'Female proportion in studies', 
            'Male proportion in studies',
            'Proportion of females in most relevant studies', 
            'Proportion of males in most relevant studies',
            'Num Studies',
            'Number of participants in most relevant studies', 
            'Number of female participants in most relevant studies',
            'Number of male participants in most relevant studies'
        ]
    )

    transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)
    transformed_df.drop(columns=['Female proportion in studies','Male proportion in studies', 'Proportion of females in most relevant studies', 'Proportion of males in most relevant studies' ],  inplace=True)
    prediction = model.predict(transformed_df)*100
    return np.around(prediction[0], 2) 


@views.route('/calculate-risk', methods=['GET', 'POST'])
def calc_risk():
    disease_prevalence = None
    prediction_risk = ""
    flash_message_risk = ""
    errorFlash = False
    # prediction_risk_male = ""
    result_string_pred = ""
    
    drug_search = ""
    disease_search = "" 

    drug_search = request.form.get('drugName')
    disease_search = request.form.get('drugCondition')
    drug_id = request.form.get('drug_id')
    from_saved_page = request.form.get('from_saved_page') 
    from_search_page = request.form.get('from_search_page')

    drugs = Drugs.query.all()
    drug_id = request.form.get('drug_id')
    # drug = Drugs.query.get(drug_id)

    if not drug_id or not drug_search or not disease_search:
        return redirect(url_for('views.home'))

    if disease_search == None or drug_search == None:
        errorFlash = True
        # flash('Please fill in all fields', 'error')
        flash_message_risk = 'Please fill in all fields'
        user_agent = request.headers.get('User-Agent').lower()
        if 'mobile' in user_agent:
            return render_template("home.html", flash_message_risk=flash_message_risk, user=current_user, errorFlash=errorFlash)    
        return render_template("home.html", flash_message_risk=flash_message_risk, user=current_user,errorFlash=errorFlash)

    drug_search = drug_search.upper()
    disease_search = disease_search.upper()

    prediction_risk = get_model(drug_search, disease_search)

    if prediction_risk is not None:
    # Convert drug_id to integer
        drug_id = int(drug_id)

    F = 0
    M = 0

    if prediction_risk is not None:
        R = 0
        M = 100 - prediction_risk
        F = prediction_risk

        if current_user.is_authenticated and current_user.sexe is not None:
            
            if current_user.sexe.lower() == 'male':

                if M > F:
                    R = M - F
                    R = str(round(R,2))
                    M = str(round(M,2))
                    result_string_pred = (
        f"The predicted risk for male patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{M}%</b>. <br><br>"
        f"Male patients have a <b>{R}%</b> lower risk of developing a reaction compared to female patients. <br>"
    )
                else:
                    R = F - M
                    R = str(round(R,2))
                    M = str(round(M,2))
                    result_string_pred = (
        f"The predicted risk for male patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{M}%</b>. <br><br>"
        f"Male patients have an additional <b>{R}%</b> risk of developing a reaction compared to female patients.<br> "
    
    )
            elif current_user.sexe.lower() == 'female':
                if F > M:
                    R = F - M
                    R = str(round(R,2))
                    F = str(round(F,2))
                    result_string_pred = (
        f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
        f"Female patients have a <b>{R}%</b> lower risk of developing a reaction compared to male patients. <br>"
        
    )
                else:
                    R = M - F
                    R = str(round(R,2))
                    F = str(round(F,2))
                    result_string_pred = (
        f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
        f"Female patients have an additional <b>{R}%</b> risk of developing a reaction compared to male patients. <br>"
    )
        else:
            if F > M:
                R = F - M
                R = str(round(R,2))
                F = str(round(F,2))
                result_string_pred = (
        f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
        f"Female patients have a <b>{R}%</b> lower risk of developing a reaction compared to male patients.<br>"
    )
            else:
                R = M - F
                R = str(round(R,2))
                F = str(round(F,2))
                result_string_pred = (
        f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
        f"Female patients have an additional <b>{R}%</b> risk of developing a reaction compared to male patients.<br>"
    )
    result_string_pred = result_string_pred 

    search_term = ""
    results=[]
       
    if prediction_risk is not None and current_user.is_authenticated:
        if from_saved_page == 'true':
            drugs = current_user.drugs
            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("saved-mobile.html", drugs=drugs, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred, result_drug_id=drug_id)
            return render_template("saved.html", drugs=drugs, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred, result_drug_id=drug_id)

        if from_search_page == 'true':
            if request.method == 'POST':
                search_term = request.form.get('query')
            else:
                search_term = request.args.get('query')

            if search_term == "":
                flash("Please enter a search term.", category='error')
                return redirect(url_for('views.home')) 
            
            search_term = str(search_term) if search_term is not None else ''
            search_term = '%' + search_term + '%'
            # results = Drugs.query.filter(Drugs.name.ilike(search_term)).all()
            results = Drugs.query.filter(or_(Drugs.name.ilike(search_term), Drugs.disease.ilike(search_term))).all()

            if not results:
            # If no exact match is found, find the closest match
                all_drugs = Drugs.query.all()
                all_drug_names = [drug.name for drug in all_drugs]
                closest_match_result = process.extractOne(search_term, all_drug_names)
                if closest_match_result:  # Check if closest_match_result is not None
                    closest_match_name = closest_match_result[0]
                    closest_match = Drugs.query.filter_by(name=closest_match_name).first()
                    if closest_match:
                        results = [closest_match]
                else:
                    flash("No drug found with that name.", category='error')
                    return redirect(url_for('views.home'))
                
            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("search_results-mobile.html", results=results, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred, result_drug_id=drug_id)
            return render_template("search_results.html", results=results, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred, result_drug_id=drug_id)
        
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("home-mobile.html", drugs=drugs, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred, result_drug_id=drug_id)
    return render_template("home.html", drugs=drugs, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred, result_drug_id=drug_id)


@views.route('/home', methods=['GET', 'POST']) 
def home():
    disease_prevalence = None
    filtered_drugs = []
    # prediction_risk_male = ""
    result_string_pred = ""
    result_drug_id = None
    drug_id = request.form.get('drug_id')
    
    if request.method == 'POST':
        drug_filter = request.form.get('drug_filter')
        drugs = Drugs.query.all()

        if drug_filter:
            drugs = Drugs.query.filter(Drugs.disease.ilike(f'%{drug_filter}%')).all()
            
        if drug_filter == "ALL":
            drugs = Drugs.query.all()

        user_agent = request.headers.get('User-Agent').lower()
        if 'mobile' in user_agent:
            return render_template("home-mobile.html", drug_id=drug_id, result_drug_id=result_drug_id, drugs=drugs, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred)
        return render_template("home.html", drug_id=drug_id, result_drug_id=result_drug_id, drugs=drugs, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred)
    
    if not Drugs.query.first():  # Check if the database is empty
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
        csv_file = os.path.join(BASE_DIR, 'final_results.csv')  # Join the base directory with the file name
        df = pd.read_csv(csv_file)  # Use the correct path to your CSV file

        image_dir = os.path.join(BASE_DIR, 'static')

        for index, row in df.iterrows():
            drug_name = row['Drug'].replace(' ', '_').replace('/', '_')  # Replace spaces and slashes with underscores
            # fig = plt.figure(facecolor=(206/255, 227/255, 234/255, 1))  # Create a new figure with a light gray background
            # y = np.array([float(row['Female proportion in studies']), float(row['Male proportion in studies'])])
            # myLabels = ['Female', 'Male']
            # mycolors = [(236/255, 142/255, 130/255, 1), (54/255, 74/255, 93/255, 1)]
            # explode = (0.1, 0)  # explode 1st slice
            # plt.pie(y, explode=explode, labels=myLabels, colors=mycolors,
            # shadow=True, startangle=140) #, textprops={'color':"white"}
            
            # image_path = os.path.join(image_dir, f'{drug_name}.png')
            # plt.savefig(image_path, facecolor=fig.get_facecolor())
            # plt.close(fig)  # Close the figure
            relative_image_path = f'static/{drug_name}.png'

            # prediction_risk = get_model(drug_name, row['Indication'])
            # prediction_risk = str(prediction_risk)
            # result_string = f"The predicted risk for an adverse reaction to {drug_name} given {row['Indication']} is {prediction_risk} %."
            # print(result_string)
            # result_strings.append(result_string)

            drug = Drugs(
                name=drug_name,
                disease=row['Indication'],
                female_ratio=row['Female proportion in studies'],
                male_ratio=row['Male proportion in studies'],
                prevalence=relative_image_path,
                path_prevalence = f"static/images/prevalence/{row['Indication'].lower()}.png",
                prevFemale = round(row['Prevalence Women'],3),
                prevMale = round(row['Prevalence Men'],3),
                prevBoth = round(row['Prevalence Both Genders'],3)
            )

            db.session.add(drug)

        db.session.commit()

    saved_drugs = current_user.drugs if current_user.is_authenticated else []  # Retrieve saved drugs if user is authenticated
    drugs = Drugs.query.all()

    # Set the is_saved attribute for each drug
    for drug in drugs:
        drug.is_saved = drug in saved_drugs

    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("home-mobile.html", drugs=drugs, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred)
    return render_template("home.html", drugs=drugs, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred)
  
@views.route('/save-drug', methods=['POST'])
@login_required
def save_drug():
    drug_id = request.form.get('drug_id')
    drug = Drugs.query.get(drug_id)

    if drug and drug not in current_user.drugs:
        drug.is_saved = True
        current_user.drugs.append(drug)
        db.session.commit()

    # drug.user_id = current_user.id
    # drug.is_saved = True
    # db.session.commit()
    return redirect(url_for('auth.saved'))

@views.route('/unsave-drug', methods=['POST'])
@login_required
def unsave_drug():
    drug_id = request.form.get('drug_id')
    drug = Drugs.query.get(drug_id)

    if drug and drug in current_user.drugs:
        drug.is_saved = False
        current_user.drugs.remove(drug)
        db.session.commit()

    # drug.user_id = None
    # drug.is_saved = False
    # db.session.commit()
    # print(drug.is_saved)
    return redirect(url_for('auth.saved'))