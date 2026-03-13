
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer

# Datetime transformer (Transaction + DOB -> features)
class DateTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='trans_date_trans_time', dob_col='dob'):
        self.datetime_col = datetime_col
        self.dob_col = dob_col

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        # Transaction datetime features
        if self.datetime_col in df:
            dt = pd.to_datetime(df[self.datetime_col], dayfirst=True)
            df['Year'], df['Month'], df['Day'], df['Hour'] = dt.dt.year, dt.dt.month, dt.dt.day, dt.dt.hour
            df['IsWeekend'] = (dt.dt.dayofweek >= 5).astype(int)
 
        # Age from DOB
        if self.dob_col in df and 'Year' in df:
            dob = pd.to_datetime(df[self.dob_col], dayfirst=True)
            df['age'] = df['Year'] - dob.dt.year

        return df 

# -----------------------------
# Dynamic numeric transformer (handles selecting numeric columns internally)
class DynamicNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, high_cat=[], low_cat=[]):
        self.high_cat = high_cat
        self.low_cat = low_cat
        self.scaler = StandardScaler()
        self.numeric_cols = []

    def fit(self, df, y=None):
        # select only numeric columns, exclude categorical
        self.numeric_cols = [
            c for c in df.columns
            if c not in self.high_cat + self.low_cat and pd.api.types.is_numeric_dtype(df[c])
        ]
        self.scaler.fit(df[self.numeric_cols].fillna(0))
        return self

    def transform(self, df):
        return self.scaler.transform(df[self.numeric_cols].fillna(0))

    def get_feature_names_out(self, input_features=None):
        return self.numeric_cols

# -----------------------------
# Column lists
numeric_features = ['amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time',
 'merch_lat', 'merch_long', 'Year', 'Month', 'Day', 'Hour', 'IsWeekend', 'age']
high_card_cat = ['merchant', 'city', 'job']
low_card_cat = ['category', 'state', 'gender']

# -----------------------------
# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('low_cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_cat),
        ('high_cat', TargetEncoder(), high_card_cat)
    ],
    remainder='drop'
)