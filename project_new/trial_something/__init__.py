import os
from flask import Flask, session
from flask_login import LoginManager, current_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_dropzone import Dropzone
from flask_session import Session

db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    Session(app)
    
    db.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, Note, Drugs, Info
    
    with app.app_context():
        #db.drop_all()
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
    @app.before_request
    def ensure_device_seen():
        if 'device_seen' not in session:
            if current_user.is_authenticated:
                logout_user()
            session['device_seen'] = True  # Mark device as seen for future requests

    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        
        print('Created Database!')