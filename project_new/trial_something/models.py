from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Drugs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100)) #medicine name
    condition = db.Column(db.String(10000)) #condition the medicine is for
    area_affected = db.Column(db.String(500)) #area the medicine is for
    description = db.Column(db.String(10000))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    ratio = db.Column(db.String(50)) #ratio female:male
    severity = db.Column(db.String(50)) #severity of the condition
    side_effects = db.Column(db.String(500)) #side effects of the medicine
    risk = db.Column(db.String(50)) #risk of the medicine


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True) #no user can have the same email as another
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    last_name = db.Column(db.String(150))
    notes = db.relationship('Note')
    age = db.Column(db.String(50))
    sexe = db.Column(db.String(50))
    drugs = db.relationship('Drugs')
    
    