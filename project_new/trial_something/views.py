from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note, Drugs
from . import db
import json
import pandas as pd
import os

views = Blueprint('views', __name__)

@views.route('/')
def opening():
    return render_template('opening.html', user=current_user)

@views.route('/home') 
def home():
    # if request.method == 'POST': 
    #     note = request.form.get('note')#Gets the note from the HTML 

    #     if len(note) < 1:
    #         flash('Note is too short!', category='error') 
    #     else:
    #         new_note = Note(data=note, user_id=current_user.id)  #providing the schema for the note 
    #         db.session.add(new_note) #adding the note to the database 
    #         db.session.commit()
    #         flash('Note added!', category='success')

    # data= pd.read_csv("information.csv")
    # name_column = data['name']
    # condition_column = data['condition']
    # area_affected_column = data['area_affected']
    # ratio_column = data['ratio']
    # severity_column = data['severity']
    # side_effects_column = data['side_effects']
    # risk_column = data['risk']
    # description_column = data['description']

    # first_row = data.iloc[1]
    # name = first_row['name_column']
    # condition = first_row['condition_column']
    # area_affected = first_row['area_affected_column']
    # ratio = first_row['ratio_column']
    # severity = first_row['severity_column']
    # side_effects = first_row['side_effects_column']
    # risk = first_row['risk_column']
    # description = first_row['description_column']
    
    # drug = Drugs.query.filter_by(name=name, condition=condition, area_affected=area_affected, ratio=ratio, 
    #     severity=severity, side_effects=side_effects, risk=risk, description=description)
    # db.session.add(drug)
    # db.session.commit()




    return render_template("home.html", user=current_user) 
   


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