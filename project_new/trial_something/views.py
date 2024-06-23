from flask import Blueprint, redirect, render_template, request, flash, jsonify, url_for
from flask_login import login_required, current_user
from .models import Note, Drugs
from . import db
import json
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import func

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


@views.route('/home', methods=['GET', 'POST']) 
def home():
    disease_prevalence = None

    if request.method == 'POST':
        drug_filter = request.form.get('drug_filter')

        if drug_filter:
            filtered_drugs = Drugs.query.filter(Drugs.disease.ilike(f'%{drug_filter}%')).all()
            
        if drug_filter == "ALL":
            filtered_drugs = Drugs.query.all()

        user_agent = request.headers.get('User-Agent').lower()
        if 'mobile' in user_agent:
            return render_template("home-mobile.html", drugs=filtered_drugs, user=current_user, disease_prevalence=disease_prevalence)
        return render_template("home.html", drugs=filtered_drugs, user=current_user, disease_prevalence=disease_prevalence)
    
    if not Drugs.query.first():  # Check if the database is empty
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
        csv_file = os.path.join(BASE_DIR, 'final_results.csv')  # Join the base directory with the file name
        df = pd.read_csv(csv_file)  # Use the correct path to your CSV file

        image_dir = os.path.join(BASE_DIR, 'static')

        for index, row in df.iterrows():
            drug_name = row['Drug'].replace(' ', '_').replace('/', '_')  # Replace spaces and slashes with underscores
            fig = plt.figure(facecolor=(206/255, 227/255, 234/255, 1))  # Create a new figure with a light gray background
            y = np.array([float(row['Female proportion in studies']), float(row['Male proportion in studies'])])
            myLabels = ['Female', 'Male']
            mycolors = [(236/255, 142/255, 130/255, 1), (54/255, 74/255, 93/255, 1)]
            explode = (0.1, 0)  # explode 1st slice
            plt.pie(y, explode=explode, labels=myLabels, colors=mycolors, autopct='%1.1f%%',
            shadow=True, startangle=140)
            
            image_path = os.path.join(image_dir, f'{drug_name}.png')
            plt.savefig(image_path, facecolor=fig.get_facecolor())
            plt.close(fig)  # Close the figure
            relative_image_path = f'static/{drug_name}.png'

            drug = Drugs(
                name=drug_name,
                disease=row['Indication'],
                female_ratio=row['Female proportion in studies'],
                male_ratio=row['Male proportion in studies'],
                prevalence=relative_image_path,
                path_prevalence = f"static/images/prevalence/{row['Indication'].lower()}.png"
            )
            db.session.add(drug)

        db.session.commit()

    drugs = Drugs.query.all()

    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("home-mobile.html", drugs=drugs, user=current_user, disease_prevalence=disease_prevalence)
    return render_template("home.html", drugs=drugs, user=current_user, disease_prevalence=disease_prevalence)

    
@views.route('/save-drug', methods=['POST'])
@login_required
def save_drug():
    drug_id = request.form.get('drug_id')
    drug = Drugs.query.get(drug_id)
    drug.is_saved = True
    db.session.commit()
    return redirect(url_for('auth.saved'))

@views.route('/unsave-drug', methods=['POST'])
@login_required
def unsave_drug():
    drug_id = request.form.get('drug_id')
    drug = Drugs.query.get(drug_id)
    drug.is_saved = False
    db.session.commit()
    return redirect(url_for('views.home'))

@views.route('/delete-note', methods=['POST'])
def delete_note():  
    note = json.loads(request.data) # this function expects a JSON from the INDEX.js file 
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})