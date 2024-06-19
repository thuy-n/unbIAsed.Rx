import os
from flask import Blueprint, jsonify, render_template, request, flash, redirect, url_for
from .models import User, Note, Drugs
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   #means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, ValidationError
from wtforms.validators import DataRequired, Email
from werkzeug.utils import secure_filename
from sqlalchemy import text, func, create_engine
from fuzzywuzzy import process


auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if email == '' or password == '':
            flash('Please fill in all fields.', category='error')
        elif user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("login.html", user=current_user)

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', category='success')
    return redirect(url_for('views.opening'))

class UpdateUserForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(message = 'Enter a valid email'),Email()])
    firstname = StringField('First Name', validators=[DataRequired(message = 'Enter your first name')])
    
    submit = SubmitField('Update!')

    def validate_email(self, email):
        if User.query.filter_by(email=email.data).first():
            raise ValidationError('This email has been registered already!')


@auth.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        try:
            user = User.query.filter_by(id=current_user.id).first()

            # Get the field and value from the form data
            field = request.form.get('field')
            value = request.form.get('value')

            # Update the selected field of the user
            if field == '1' and value:  # First Name
                user.first_name = value
            elif field == '2' and value:  # Last Name
                user.last_name = value
            elif field == '3' and value:  # Age
                user.age = value
            elif field =='4' and value: #Sexe
                value = value.capitalize()
                if value == 'Female' or value == 'Male':
                    user.sexe = value
                elif value != 'Female' or value != 'Male':
                    flash('Please enter a valid answer for the sexe field', category='error')


            # Commit the changes to the database
            db.session.commit()

        except Exception as e:
            print("Error updating user age: ", e)

    return render_template("profile.html", user=current_user)

@auth.route('/saved')
@login_required
def saved():
    drugs = Drugs.query.filter_by(is_saved=True).all()
    return render_template("saved.html", drugs=drugs, user=current_user)
 
@auth.route('/about')
def about():
    return render_template('about.html', user=current_user)

@auth.route('/settings')
def settings():
    return render_template('settings.html', user=current_user)


@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        last_name = request.form.get('lastName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = User(email=email, first_name=first_name, last_name=last_name, password=generate_password_hash(
                password1, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            flash('Account created!', category='success')
            return redirect(url_for('views.home'))

    return render_template("sign_up.html", user=current_user)

@auth.route('/delete_account', methods=['GET','POST'])
@login_required
def delete_account():
    #go look at Modal (bootstrap) for this
    try:
        user = User.query.filter_by(id=current_user.id).first()

        # Delete the user
        db.session.delete(user)

        # Commit the changes to the database
        db.session.commit()

        # Log out the user
        logout_user()

        flash("Your account has been deleted successfully.")
        return redirect(url_for('views.opening'))

    except Exception as e:
        print("Error deleting user account: ", e)
        flash("An error occurred while trying to delete your account.")
        return redirect(url_for('auth.profile'))



@auth.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_term = request.form.get('query')
    else:
        search_term = request.args.get('query')

    if search_term == "":
        flash("Please enter a search term.", category='error')
        return redirect(url_for('views.home'))

    # Perform a case-insensitive pattern search
    search_term = '%' + search_term + '%'
    results = Drugs.query.filter(Drugs.name.ilike(search_term)).all()

    if not results:
        # If no exact match is found, find the closest match
        all_drugs = Drugs.query.all()
        all_drug_names = [drug.name for drug in all_drugs]
        closest_match_name = process.extractOne(search_term, all_drug_names)[0]
        closest_match = Drugs.query.filter_by(name=closest_match_name).first()
        if closest_match:
            results = [closest_match]
        else:
            flash("No drug found with that name.", category='error')
            return redirect(url_for('views.home'))

    return render_template("search_results.html", results=results, user=current_user)
   
    # search_term = None
    # if request.method == 'POST':
    #     search_term = request.form.get('query')
    # else:
    #     search_term = request.args.get('query')

    # if search_term == "":
    #     flash("Please enter a search term.", category='error')
    #     return redirect(url_for('views.home'))

    
    # results = Drugs.query.filter(Drugs.name.contains(search_term)).all()

    # if not results:
    #     flash("No drug found with that name.", category='error')
    #     return redirect(url_for('views.home'))

    # return render_template("search_results.html", results=results, user=current_user)


@auth.route('/identify', methods=['GET','POST'])
def identify():
    if request.method == 'POST':
        image_file = request.files.get('uploaded-image')
        label_file = request.files.get('uploaded-label')
        button_clicked = request.form.get('submit-button')

        # Check if at least one file was uploaded
        if image_file is None and label_file is None:
            flash('Either an image or a label file must be uploaded', 'error')
            return render_template("identify.html", user=current_user)

        # Process the image file if the 'image' button was clicked and a file was uploaded
        if button_clicked == 'image' and image_file and image_file.filename != '':
            image_filename = secure_filename(image_file.filename)
            image_filepath = os.path.join('/tmp', image_filename)
            image_file.save(image_filepath)
            # Add your image processing code here
            flash('Image successfully identified', 'success')


        # Process the label file if the 'label' button was clicked and a file was uploaded
        elif button_clicked == 'label' and label_file and label_file.filename != '':
            label_filename = secure_filename(label_file.filename)
            label_filepath = os.path.join('/tmp', label_filename)
            label_file.save(label_filepath)
            # Add your label processing code here
            flash('Label successfully identified', 'success')


        else:
            flash('No file was uploaded', 'error')

        return render_template("identify.html", user=current_user)

    else:
        return render_template("identify.html", user=current_user)