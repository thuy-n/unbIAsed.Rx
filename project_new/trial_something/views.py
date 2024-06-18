from flask import Blueprint, redirect, render_template, request, flash, jsonify, url_for
from flask_login import login_required, current_user
from .models import Note, Drugs
from . import db
import json
import pandas as pd
import os

views = Blueprint('views', __name__)

@views.route('/')
def opening():
    logged_in = False
    if current_user.is_authenticated:
        logged_in = True
    return render_template('opening.html', logged_in=logged_in, user=current_user)

@views.route('/home') 
def home():
    if not Drugs.query.first():  # Check if the database is empty
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
        csv_file = os.path.join(BASE_DIR, 'drugs.csv')  # Join the base directory with the file name
        df = pd.read_csv(csv_file)  # Use the correct path to your CSV file
        print(df.columns)
        for index, row in df.iterrows():
            drug = Drugs(
                name=row['name'],
                condition=row['condition'],
                area_affected=row['area_affected'],
                ratio=row['ratio'],
                severity=row['severity'],
                side_effects=row['side_effects'],
                risk=row['risk']
            )
            db.session.add(drug)

        db.session.commit()

    drugs = Drugs.query.all()
    return render_template("home.html", drugs=drugs, user=current_user)

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