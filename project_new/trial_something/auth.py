import os
import re
from flask import Blueprint, jsonify, render_template, request, flash, redirect, session, url_for
import pandas as pd
from .models import User, Drugs
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   #means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, ValidationError
from wtforms.validators import DataRequired, Email
from werkzeug.utils import secure_filename
from fuzzywuzzy import process
import pytesseract
import cv2
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
import difflib
from sqlalchemy import or_
import torch
import torch.nn as nn
import torch
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
import timm
from albumentations import (
    Compose, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    Rotate, ShiftScaleRotate, Transpose
)
from albumentations.pytorch import ToTensorV2
from trial_something.views import get_model

class BaselineModel(nn.Module):
    def __init__(self, num_classes: int, model_type = 'shuffleNet'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'shuffleNet':
          self.model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
          self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_type == 'ResNext':
          self.model = timm.create_model('resnext50_32x4d', pretrained=True)
          n_features = self.model.fc.in_features
          self.model.fc = nn.Linear(n_features, num_classes)
        elif model_type == 'resnet50':
          backbone = timm.create_model(model_type, pretrained=True)
          n_features = backbone.fc.in_features
          self.backbone = nn.Sequential(*backbone.children())[:-2]
          self.classifier = nn.Linear(n_features, num_classes)
          self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
      if self.model_type != 'resnet50':
        return self.model(x)
      elif self.model_type == 'resnet50':
        x = self.backbone(x)
        feats = x
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def load(filename: str):
        checkpoint = torch.load(filename)
        model = BaselineModel(checkpoint["num_classes"])
        model.load_state_dict(checkpoint["params"])
        return model

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
                login_user(user, remember=True)
                return redirect(url_for('views.home'))
            elif not check_password_hash(user.password, password):
                flash('Incorrect password, try again.', category='error')

                return redirect(url_for('auth.login'))
            else:
                flash('Email does not exist.', category='error')
                return redirect(url_for('auth.login'))
        else:
            flash('Email does not exist.', category='error')

    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("login-mobile.html", user=current_user)
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

            # Commit the changes to the database
            db.session.commit()

            if field == '3' and value:  # Age
                user.age = value
            elif field == '2' and value:  # Last Name
                user.last_name = value
            elif field == '1' and value:  # First Name
                user.first_name = value
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

    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("profile-mobile.html", user=current_user)
    return render_template("profile.html", user=current_user)

@auth.route('/saved')
@login_required
def saved():
    drugs = Drugs.query.filter_by(is_saved=True, user_id=current_user.id).all() #user_id=current_user.id
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("saved-mobile.html", drugs=drugs, user=current_user)
    return render_template("saved.html", drugs=drugs, user=current_user)
 
@auth.route('/about')
def about():
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template('about-mobile.html', user=current_user)
    return render_template('about.html', user=current_user)

# @auth.route('/settings')
# def settings():
    return render_template('settings.html', user=current_user)

@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName').capitalize()
        last_name = request.form.get('lastName').capitalize()
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        sex = request.form.get('sexSelect', '').strip()

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
            new_user = User(email=email, sexe = None, first_name=first_name, last_name=last_name, password=generate_password_hash(
                password1, method='pbkdf2:sha256'))
            
            if sex == 'female' or sex == 'male':
                new_user.sexe = sex.capitalize()
            elif sex =='none':
                new_user.sexe = None
            
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            flash('Account created!', category='success')
            return redirect(url_for('views.home'))
            
            

    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("sign_up-mobile.html", user=current_user)
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
        return render_template("search_results-mobile.html", results=results, user=current_user)
    return render_template("search_results.html", results=results, user=current_user)

def preprocess(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Convert to uppercase
    sentence = sentence.lower()
    # Tokenize
    words = sentence.split()
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Join words back into sentence
    sentence = ' '.join(words)
    return sentence

def predict_with_cnn(image_path, model_path, input_width, input_height):
    # Define the transformation chain
    transform = Compose([
        RandomResizedCrop(input_height, input_width),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    # print('Processing image')
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Load image and preprocess
    image = cv2.imread(image_path)
    augmented = transform(image=image)
    image = augmented['image']  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    # print('Done processing image')

    # Load the model
    model = BaselineModel(10, 'resnet50')
    # model = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    # Make prediction
    with torch.no_grad():
        output = model(image.to(device))

    # Interpret results as classification
    probs = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return predicted_class


@auth.route('/identify', methods=['GET','POST'])
def identify():
    sentence = ""
    text=""
    word=None
    something=""
    prediction_risk = ""
    result_string = ""
    errorFlash = False
    flash_message = ""
    flash_message_label = ""
    flash_message_pill = ""
    flash_message_risk = ""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    csv_file = os.path.join(BASE_DIR, 'final_results.csv')  # Join the base directory with the file name
    df = pd.read_csv(csv_file)
    df.iloc[1:,0] = df.iloc[1:,0].str.lower()
    meds = df.iloc[1:,0].tolist()
    
    
    if request.method == 'POST':
        
        image_file = request.files.get('uploaded-pill-image')
        label_file = request.files.get('uploaded-label-image')
        button_clicked1 = request.form.get('submit-button1')
        button_clicked2 = request.form.get('submit-button2')
        drug_search = ""
        drug_menu = request.form.get('drugRiskSelect')

        if drug_menu == 'input':
            drug_search = request.form.get('drugRiskInput')
        elif drug_menu == 'select':
            drug_search = request.form.get('selectDrug')
        # else:
        #     flash('Please select a valid option', 'error')
        #     user_agent = request.headers.get('User-Agent').lower()
        #     if 'mobile' in user_agent:
        #         return render_template("identify-mobile.html", user=current_user,meds=meds)
        #     return render_template("identify.html", user=current_user,meds=meds)

        disease_search = request.form.get('diseaseRisk')
        button_clicked3 = request.form.get('submit-button3')

        # Check if at least one file was uploaded
        # if image_file is None and label_file is None: #and drug_search == '' and disease_search == ''
        #     errorFlash = True
        #     # flash('Either an image or a label file must be uploaded', 'error')
        #     flash_message = 'Either an image or a label file must be uploaded'
        #     flash_message_label = flash_message
        #     flash_message_pill = flash_message  

        #     user_agent = request.headers.get('User-Agent').lower()
        #     if 'mobile' in user_agent:
        #         return render_template("identify-mobile.html", flash_message_pill = flash_message_pill,  flash_message_label = flash_message_label, user=current_user,meds=meds, errorFlash=errorFlash)  
        #     return render_template("identify.html", flash_message_pill = flash_message_pill, flash_message_label = flash_message_label, user=current_user,meds=meds, errorFlash=errorFlash)

        if image_file is None: #pill
            errorFlash = True
            flash_message_pill = 'An image file must be uploaded'

            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("identify-mobile.html", flash_message_pill = flash_message_pill, user=current_user,meds=meds, errorFlash=errorFlash)  
            return render_template("identify.html", flash_message_pill = flash_message_pill, user=current_user,meds=meds, errorFlash=errorFlash)

        if label_file is None: #label
            errorFlash = True
            flash_message_label = 'A label file must be uploaded'

            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("identify-mobile.html", flash_message_label = flash_message_label, user=current_user,meds=meds, errorFlash=errorFlash)
            return render_template("identify.html", flash_message_label = flash_message_label, user=current_user,meds=meds, errorFlash=errorFlash)

        # Process the image file if the 'image' button was clicked and a file was uploaded
        if button_clicked1 == 'label' and label_file and label_file.filename != '':
            label_filename = secure_filename(label_file.filename)
            label_filepath = os.path.join('/tmp', label_filename)
            label_file.save(label_filepath)
            
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            img = cv2.imread(label_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes = pytesseract.image_to_data(img)

            for i, box in enumerate(boxes.splitlines()):
                if i==0:
                    continue
                box = box.split()
                if len(box) == 12:
                    sentence += box[11] + " "

            session["sentence"] = sentence

            sentence = preprocess(sentence)
            
            #read csv
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
            csv_file = os.path.join(BASE_DIR, 'final_results.csv')  # Join the base directory with the file name
            df = pd.read_csv(csv_file)
            df.iloc[1:,0] = df.iloc[1:,0].str.lower()
            meds = df.iloc[1:,0].tolist()

            for word in sentence.split():
                close_matches = difflib.get_close_matches(word, meds, n=1, cutoff=0.9)
                if close_matches:
                    # flash('Label successfully identified', 'success')
                    text = close_matches[0] + ' has been found!'
                    something = close_matches[0]
                    break
                    
                else:
                    text = 'This drug is not currently in our database.'

            if sentence == "":
                errorFlash = True
                # flash('No text was found in the image', 'error')
                flash_message_label = 'No text was found in the image'
                
            
            # os.remove(label_filepath)


        # Process the label file if the 'label' button was clicked and a file was uploaded
        elif button_clicked2 == 'pill' and image_file and image_file.filename != '':
            image_filename = secure_filename(image_file.filename)
            image_filepath = os.path.join('/tmp', image_filename)
            image_file.save(image_filepath)

            # Define the class dictionary
            class_dict = {'Alaxan': 0, 'Bactidol': 1, 'Biogesic': 2, 'Lamictal': 4, 'DayZinc': 3, 'Rivaroxaban': 5,
                'Fish Oil': 6, 'Kremil S': 7, 'Medicol': 8, 'Neozep': 9}
            
            # Create a reverse dictionary
            reverse_class_dict = {v: k for k, v in class_dict.items()}

            # Example usage
            #model_path = r'C:\Users\anbgo\coding_projects_flask\GitHub-Rx\project_new\trial_something\resnet50-2.pt'
            model_path = os.path.join(BASE_DIR, 'resnet50-2.pt')
            input_width = 224  # Replace with your model's input width
            input_height = 224  # Replace with your model's input height
            predicted_class = predict_with_cnn(image_filepath, model_path, input_width, input_height)
            predicted_class_name = reverse_class_dict[predicted_class]
            # print(f"Predicted class: {predicted_class_name}")
            pill = predicted_class_name + " has been found!"

            # flash('Pill successfully identified', 'success')
            # os.remove(image_filepath)

        
            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("identify-mobile.html", user=current_user, text=text, word=word, something=something, pill=pill,meds=meds, errorFlash=errorFlash) 
            return render_template("identify.html", user=current_user, text=text, word=word, something=something, pill=pill,meds=meds, errorFlash=errorFlash)
        
        elif button_clicked3 == 'risk':
            drug_search = drug_search.upper()
            disease_search = disease_search.upper()

            if disease_search == None or drug_search == None:
                errorFlash = True
                # flash('Please fill in all fields', 'error')
                flash_message_risk = 'Please fill in all fields'
                user_agent = request.headers.get('User-Agent').lower()
                if 'mobile' in user_agent:
                    return render_template("identify-mobile.html", flash_message_risk=flash_message_risk, user=current_user, text=text, word=word, something=something, meds=meds, errorFlash=errorFlash)    
                return render_template("identify.html", flash_message_risk=flash_message_risk, user=current_user, text=text, word=word, something=something, meds=meds, errorFlash=errorFlash)
                    
            if disease_search == 'SELECT CONDITION':
                errorFlash = True
                # flash('Please select a valid condition', 'error')
                flash_message_risk = 'Please select a valid condition'
                user_agent = request.headers.get('User-Agent').lower()
                if 'mobile' in user_agent:
                    return render_template("identify-mobile.html", flash_message_pill=flash_message_pill, flash_message_label=flash_message_label, flash_message_risk=flash_message_risk, errorFlash=errorFlash, user=current_user, text=text, word=word, something=something, meds=meds)    
                return render_template("identify.html", flash_message_pill=flash_message_pill, flash_message_label=flash_message_label, flash_message_risk=flash_message_risk, errorFlash=errorFlash, user=current_user, text=text, word=word, something=something, meds=meds)

            prediction_risk = get_model(drug_search, disease_search)

            # After extracting or validating prediction_risk
            if prediction_risk is not None:
                # Convert prediction_risk to string
                if current_user.is_authenticated and current_user.sexe is not None:
                    if current_user.sexe.lower() == 'male':
                        prediction_risk = 100 - prediction_risk
                
                prediction_risk = str(prediction_risk)
                
            # Now prediction_risk is guaranteed to be a string when constructing result_string
            result_string = f"The predicted risk for an adverse reaction to {drug_search} given {disease_search} is {prediction_risk} %."

            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("identify-mobile.html", errorFlash=errorFlash, user=current_user, text=text, word=word, something=something, result_string=result_string,meds=meds) 
            return render_template("identify.html", errorFlash=errorFlash, user=current_user, text=text, word=word, something=something, result_string=result_string,meds=meds)
            
        # else:
        #     if button_clicked1 == 'label' and not label_file:
        #         errorFlash = True
        #         flash_message_label = 'No file was uploaded'
        #         # flash('No file was uploaded', 'error')
        #     if button_clicked2 == 'pill' and not image_file:
        #         errorFlash = True
        #         flash_message_pill = 'No file was uploaded'

                # flash('No file was uploaded', 'error')
            # if drug_search == '' or disease_search == '':
            #     flash('Please fill in all fields', 'error')

        user_agent = request.headers.get('User-Agent').lower()
        if 'mobile' in user_agent:
            return render_template("identify-mobile.html", errorFlash=errorFlash, user=current_user, text=text, word=word, something=something,meds=meds)    
        return render_template("identify.html", errorFlash=errorFlash, user=current_user, text=text, word=word, something=something,meds=meds)

    else:
        user_agent = request.headers.get('User-Agent').lower()
        if 'mobile' in user_agent:
            return render_template("identify-mobile.html", errorFlash=errorFlash, user=current_user, word=word, something=something,meds=meds)   
        return render_template("identify.html", errorFlash=errorFlash, user=current_user, word=word, something=something,meds=meds)

@auth.route('/learn')
def learn():
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template('learn-mobile.html', user=current_user)
    return render_template('learn.html', user=current_user)