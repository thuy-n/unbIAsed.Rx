from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

# Association table for the many-to-many relationship between User and Drug
user_drug = db.Table('user_drug',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('drug_id', db.Integer, db.ForeignKey('drugs.id'), primary_key=True)
)

class Info(db.Model):
    #info for studies
    id = db.Column(db.Integer, primary_key=True)
    NCT = db.Column(db.String(50), unique=True) #NCT number
    num_men = db.Column(db.Integer) #number of
    num_women = db.Column(db.Integer)
    num_participants = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Drugs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False) #medicine name, must have something
    disease = db.Column(db.String(100)) #disease the medicine is for, disease identification
    female_ratio = db.Column(db.String(50)) 
    male_ratio = db.Column(db.String(50))
    prevalence = db.Column(db.String(500)) #prevelence of the disease, path to image
    path_prevalence = db.Column(db.String(500)) #path to the image
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    is_saved = db.Column(db.Boolean, default=False)
    prevFemale = db.Column(db.String(50))
    prevMale = db.Column(db.String(50))
    prevBoth = db.Column(db.String(50))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True) #no user can have the same email as another
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    last_name = db.Column(db.String(150))
    notes = db.relationship('Note')
    age = db.Column(db.String(50))
    sexe = db.Column(db.String(50))
    drugs = db.relationship('Drugs', secondary=user_drug, backref=db.backref('users', lazy='dynamic'))
    info = db.relationship('Info', backref='user', lazy=True)
    
    