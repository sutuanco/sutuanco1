from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class CheckForm(FlaskForm):
    linkShopee = StringField('Link Shopee', validators=[DataRequired()])
    submit = SubmitField('FIND')