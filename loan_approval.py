import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import math

#----------------------DATA PREPROCESSING----------------------

credit_history = pd.read_csv('credit_approval_data/credit_record.csv')
app_data = pd.read_csv('credit_approval_data/application_record.csv')

def IV(data, col):#Will use later to check on usefull columns
  WOE = pd.DataFrame()

  bad_customers = data['CREDIT_STATUS'].sum()
  good_customers = data.shape[0] - bad_customers

  WOE['BAD'] = data.groupby(col)['CREDIT_STATUS'].sum() / bad_customers
  WOE['GOOD'] = (data.groupby(col)['CREDIT_STATUS'].count() - data.groupby(col)['CREDIT_STATUS'].sum()) / good_customers#Before minus sign we found total number of each class in column and after exclude bad customers from it
  # Add a small constant to avoid division by zero
  epsilon = 1e-10
  WOE['WOE'] = np.log((WOE['GOOD'] + epsilon) / (WOE['BAD'] + epsilon))
  IV = (WOE['WOE'] * (WOE['GOOD'] - WOE['BAD'])).sum()
  
  return IV   

class NoLoanDropper(BaseEstimator, TransformerMixin):
  
  def __init__(self):
    pass
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, credit_data):
    credit_data = credit_data[credit_data['STATUS'] != 'X']
    
    return credit_data

class ReassigningSortingValues(BaseEstimator, TransformerMixin):
  
  def __init__(self):
    pass
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, credit_data):
    
    for _ in range(6):
  
      if _ < 3:
        credit_data = credit_data.replace({'STATUS': {str(_): 'B'}})#Acceptable default
      
      else:
        credit_data = credit_data.replace({'STATUS': {str(_): 'A'}})#Default
    
    credit_data = credit_data.sort_values(by=['ID', 'STATUS'])#Now all A values are in top of the list in STATUS column for each id. So there're no missed defaults for each id

    return credit_data

class DataAbsoluteValues(BaseEstimator, TransformerMixin):
  
  def __init__(self, features=['DAYS_EMPLOYED', 'DAYS_BIRTH']):
    self.features = features
    
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    data[self.features] = abs(data[self.features])
    
    return data

class DuplicatesDropper(BaseEstimator, TransformerMixin):
  
  def __init__(self):
    pass
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    data = data.drop_duplicates(subset=data.columns.to_list().remove('ID'), keep='first')
    
    return data

class OutlierHandler(BaseEstimator, TransformerMixin):
  
  def __init__(self, names=['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS']):
    self.names = names
    
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    #handle outlier for pensioners(1000 years employed replace with 50)
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].replace(1000, 50)
    #calcualate IQR
    Q1 = data[self.names].quantile(0.25)
    Q3 = data[self.names].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 3.3 * IQR
    upper_limit = Q3 + 3.3 * IQR

    outlier_mask = (data[self.names] < lower_limit) | (data[self.names] > upper_limit)#names marked like True
    data[self.names] = data[self.names].where(~outlier_mask, np.nan)#This method replaces values where the condition is False with np.nan
    data = data.dropna(subset=self.names, axis=0)

    return data

class DaysToYearsEncoder(BaseEstimator, TransformerMixin):
  
  def __init__(self, features=['DAYS_EMPLOYED', 'DAYS_BIRTH']):
    self.features = features
    
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    
    for _ in self.features:
      data[_] = data[_].apply(lambda x: math.floor(x / 365))
          
    return data

class PerformanceWindow(BaseEstimator, TransformerMixin):
  
  def __init__(self, credit_data):
    self.credit_data = credit_data
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    ids = self.credit_data.groupby('ID', as_index=False)['ID'].value_counts()
    ids = ids[ids['count'] >= 12]
    common_ids = ids['ID'].to_list()
    data = data[data['ID'].isin(common_ids)].reset_index(drop=True)
    
    return data

#Determine IDs with good and bad behavior
#According to Vintage Analysis, if customer defaults (90 days or more past due) during the performance window, borrower would be considered as a 'bad' customer
class Behavior(BaseEstimator, TransformerMixin):
  
  def __init__(self, credit_data):
    self.credit_data = credit_data
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    data['CREDIT_STATUS'] = 0
    filtered_ids = set(self.credit_data.loc[self.credit_data['STATUS'] == 'A']['ID'])
    data.loc[data['ID'].isin(filtered_ids), 'CREDIT_STATUS'] = 1#All ids with 90+ default are bad now
    
    return data

#Let's add the number of defaults to app_data from credit_history
class TotalDefaults(BaseEstimator, TransformerMixin):
  
  def __init__(self, credit_data):
    self.credit_data = credit_data
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    defaults_number = self.credit_data.groupby('ID')['STATUS'].sum().apply(lambda x: x[0:12]).reset_index()
    defaults_number['STATUS'] = defaults_number['STATUS'].apply(lambda x: x.count('B') + x.count('A'))
    defaults_number = defaults_number.rename(columns={'STATUS': 'TOTAL_DEFAULTS'})
    data = data.merge(defaults_number, on='ID', how='right')
    
    return data

class ColumnMissingValueFiller(BaseEstimator, TransformerMixin):
  
  def __init__(self):
    pass
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    data.loc[data['NAME_INCOME_TYPE'] == 'Pensioner', 'OCCUPATION_TYPE'] = 'Pensioner'
    data = data.dropna(axis=0)
    
    return data

class FeatureDropper(BaseEstimator, TransformerMixin):
  
  def __init__(self, features=['FLAG_MOBIL', 'FLAG_OWN_CAR', 'FLAG_WORK_PHONE', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS']):
    self.features = features
    
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    data = data.drop(columns=self.features)

    return data

class InstancesMixer(BaseEstimator, TransformerMixin):
  
  def __init__(self):
    pass
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    data = data[data['NAME_INCOME_TYPE'] != 'Student'].reset_index(drop=True)
    #mix Academic degree with Higher education and Lower secondary with Secondary / secondary special
    data['NAME_EDUCATION_TYPE'] = data['NAME_EDUCATION_TYPE'].replace({'Higher education': 'Higher / Academic', 
                                                                       'Academic degree': 'Higher / Academic', 
                                                                       'Lower secondary': 'Secondary / secondary special'})
    data['NAME_HOUSING_TYPE'] = data['NAME_HOUSING_TYPE'].replace({'Rented apartment': 'Rented / Officce / Co-op', 
                                                                   'Office apartment': 'Rented / Officce / Co-op',
                                                                   'Co-op apartment': 'Rented / Officce / Co-op'})
    data['OCCUPATION_TYPE'] = data['OCCUPATION_TYPE'].replace({'Cooking staff': 'Service and Support Staff',
                                                               'Security staff': 'Service and Support Staff',
                                                               'Cleaning stuff': 'Service and Support Staff',
                                                               'Private service staff': 'Service and Support Staff',
                                                               'Low-skill Laborers': 'Service and Support Staff',
                                                               'Secretaries': 'Service and Support Staff',
                                                               'Waiters/barmen staff': 'Service and Support Staff',
                                                               'HR staff': 'Service and Support Staff',
                                                               'IT staff': 'High skill tech staff',
                                                               'Realty agents': 'Managers',
                                                               'Cleaning staff': 'Service and Support Staff'})
    
    return data

class CatEncoder(BaseEstimator, TransformerMixin):
  
  def __init__(self, 
               ohe=['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], 
               oe=['CODE_GENDER','NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE'],
               num_columns=['ID', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'TOTAL_DEFAULTS', 'CREDIT_STATUS']):
    self.ohe = ohe
    self.oe = oe
    self.num_columns = num_columns
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    ohe_encoder = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
    oe_encoder = OrdinalEncoder().set_output(transform='pandas')
    
    ohe_transformed = ohe_encoder.fit_transform(data[self.ohe])
    oe_transformed = oe_encoder.fit_transform(data[self.oe])
    
    data = pd.concat([ohe_transformed, oe_transformed, data[self.num_columns]], axis=1)
    
    return data
  
class Scaler(BaseEstimator, TransformerMixin):
  
  def __init__(self, features=['AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'DAYS_BIRTH']):
    self.features = features
    
  def fit(self, X, y=None):
    return self
  
  def transform(self, data):
    scaler = MinMaxScaler()
    data[self.features] = scaler.fit_transform(data[self.features])
        
    return data

def PreprocessingPipeline(data, credit_data):
    credit_pipeline = Pipeline([
        ('noloandropper', NoLoanDropper()),
        ('reassigningsortingvalues', ReassigningSortingValues())
    ])
    credit_data_prep = credit_pipeline.fit_transform(credit_data)
    
    data_pipeline = Pipeline([
        ('dataabsolutevalues', DataAbsoluteValues()),
        ('duplicatesdropper', DuplicatesDropper()),
        ('outlierhandler', OutlierHandler()),
        ('daystoyearsencoder', DaysToYearsEncoder()),
        ('performancewindow', PerformanceWindow(credit_data_prep)),
        ('behavior', Behavior(credit_data_prep)),
        ('totaldefaults', TotalDefaults(credit_data_prep)),
        ('columnmissingvaluesfiller', ColumnMissingValueFiller()),
        ('featuredropper', FeatureDropper()),
        ('instancesmixer', InstancesMixer()),
        ('catencoder', CatEncoder()),
        ('scaler', Scaler())
    ])
    data_transformed = data_pipeline.fit_transform(data, credit_data_prep)

    return data_transformed

df_prep = PreprocessingPipeline(app_data, credit_history)

#----------------------STREAMLIT----------------------

import streamlit as st
import joblib
import time

st.write("""
# Credit Approval Forecasting
This app is using trained machine learning model to predict loan approval based on person's data. If you do not have any income, then you are denied automatically. The app assumes that you already have a credit history of at least 12 months. Fill in the fields below and click Predict.

Some characteristics may be missing, for example, you work as a software engineer, but there is no this profession in the occupation type menu, then you should choose the most similar activity to yours.
""")

st.write("""
## Gender
""")
gender_boxes = ['M', 'F']
gender = st.selectbox('Select your gender', gender_boxes, label_visibility='collapsed')

st.write("""
## Income type
""")
income_type_boxes = ['Commercial associate', 'Pensioner', 'State servant', 'Working']
income_type = st.selectbox('Select your income type', income_type_boxes, label_visibility='collapsed')

st.write("""
## Occupation type
""")
occupation_type_boxes = [
  'Laborers',
  'Pensioner',
  'Core staff',
  'Sales staff',
  'Managers',
  'Drivers',
  'High skill tech staff',
  'Medicine staff',
  'Accountants',
  'Cooking staff',
  'Cleaning staff',
  'Private service staff',
  'Low-skill Laborers',
  'Secretaries',
  'Waiters/barmen staff',
  'HR staff',
  'IT staff',
  'Realty agents'
]
occupation_type = st.selectbox('Select your occupation type', occupation_type_boxes, label_visibility='collapsed')

st.write("""
## Education type
""")
education_type_boxes = ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']
education_type = st.selectbox('Select your education type', education_type_boxes, label_visibility='collapsed')

st.write("""
## Family status
""")
family_status_boxes = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
family_status = st.selectbox('Select your family status', family_status_boxes, label_visibility='collapsed')

st.write("""
## Housing type
""")
housing_type_boxes = ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment']
housing_type = st.selectbox('Select your housing type', housing_type_boxes, label_visibility='collapsed')

st.write("""
## Income
""")
income = int(st.text_input('Your year income in USD', 0))#fix this

st.write("""
## Years birth
""")
years_birth = st.slider('How old are you?', 21, 68, 21, label_visibility='collapsed')

st.write("""
## Years employed
If you are a pensioner, then set the value to 50 on the slider
""")
years_employed = st.slider('How many years are you employed in the company?', 0, 50, 0)

st.write("""
## Defaults
How many defaults do you have for a loan term of at least 12 months? If there are more than 12 defaults, then set the value to 12 on the slider.
""")
defaults = st.slider('How many defaults do you have?', 0, 12, 0)

data = {
  'Selected Value': [gender, income_type, occupation_type, 
                     education_type, family_status, housing_type,
                     income, years_birth, years_employed, defaults]
}
index_labels = ['Gender', 'Income type', 'Occupation type',
                'Education type', 'Family status', 'Housing type',
                'Income', 'Years birth', 'Years employed', 'Defaults']
df = pd.DataFrame(data, index=index_labels)

st.write("""
## Check your data for mistakes
""")
st.write(df.T)

button = st.button('Predict')

profile_data = {'ID': [0],
                'CODE_GENDER': [gender],
                'FLAG_OWN_CAR': [0],
                'FLAG_OWN_REALTY': [0],
                'CNT_CHILDREN': [0],
                'AMT_INCOME_TOTAL': [income],
                'NAME_INCOME_TYPE': [income_type],
                'NAME_EDUCATION_TYPE': [education_type],
                'NAME_FAMILY_STATUS': [family_status],
                'NAME_HOUSING_TYPE': [housing_type],
                'DAYS_BIRTH': [years_birth * 365],
                'DAYS_EMPLOYED': [years_employed * 365],
                'FLAG_MOBIL': [0],
                'FLAG_WORK_PHONE': [0],
                'FLAG_PHONE': [0],
                'FLAG_EMAIL': [0],
                'OCCUPATION_TYPE': [occupation_type],
                'CNT_FAM_MEMBERS': [0]
                }

profile_credit = {'ID': [0] * 12,
                  'MONTH_BALANCE': [int(i) for i in range(0, 12)],
                  'STATUS': (['0'] * defaults) + (['C'] * (12 - defaults))#calculates defaults for current profile
                  }

profile_data_df, profile_credit_df = pd.DataFrame(profile_data), pd.DataFrame(profile_credit)
app_data, credit_history = pd.concat([app_data, profile_data_df], ignore_index = True), pd.concat([credit_history, profile_credit_df], ignore_index = True)
crutch_df = PreprocessingPipeline(app_data, credit_history)
crutch_df = crutch_df.loc[crutch_df['ID'] == 0]
data_to_predict = crutch_df.drop(columns={'ID', 'CREDIT_STATUS'}, axis=1)

model = joblib.load('gb_clf_model.pkl')

if button:
  
  with st.spinner('Wait for it...'):
    time.sleep(5)
    
  forecast_result = model.predict(data_to_predict)
  
  if int(forecast_result) == 0:
    st.success("""
               ## Exciting news - your loan application has been approved!
               """)
  
  elif int(forecast_result) == 1:
    st.error("""
             ## Update on your loan application - unfortunately, it wasn't approved this time.
             """)