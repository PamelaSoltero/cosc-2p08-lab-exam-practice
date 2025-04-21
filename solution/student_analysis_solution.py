import pandas as pd
import numpy as np

class StudentAnalysis:
    def __init__(self, filepath):
        """
        Initializes the StudentAnalysis class by loading the dataset.
        """
        self.df = pd.read_csv(filepath)

    """ Data Cleaning
    - Drop the 'student_id' column as it's just an identifier
    - Fill missing values in numerical columns ('age', 'exam_score', 'study_hours_per_day', 'social_media_hours', 'netflix_hours') with their respective medians
    - Fill missing values in 'parental_education_level' with the most common education level
    - Convert Yes/No columns ('part_time_job', 'extracurricular_participation') to 1/0
    """
    def clean_data(self):
        # Drop student_id column
        self.df = self.df.drop('student_id', axis=1)
        
        # Fill missing values in numerical columns with their respective medians
        numerical_cols = ['age', 'exam_score', 'study_hours_per_day', 'social_media_hours', 'netflix_hours']
        for col in numerical_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Fill missing values in categorical column with mode
        self.df['parental_education_level'] = self.df['parental_education_level'].fillna(self.df['parental_education_level'].mode()[0])
        
        # Convert Yes/No columns to 1/0
        yes_no_cols = ['part_time_job', 'extracurricular_participation']
        for col in yes_no_cols:
            self.df[col] = self.df[col].map({'Yes': 1, 'No': 0})

    """ Data Transformation
    - Create a new column 'total_leisure_hours' that combines time spent on social media and Netflix
    - Create a new column 'study_efficiency' that measures how well study time translates to exam performance (set to 0 if study hours is 0)
    - Create a new column 'age_group' that categorizes students into three age ranges: '<18', '18-21', '22+'
    - Create a new column 'parental_education_numeric' that converts education levels into numerical values: 'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3
    """
    def transform_data(self):
        """Data Transformation
        - Create a new column 'total_leisure_hours' that combines time spent on social media and Netflix
        - Create a new column 'study_efficiency' that measures how well study time translates to exam performance (set to 0 if study hours is 0)
        - Create a new column 'age_group' that categorizes students into three age ranges: '<18', '18-21', '22+'
        - Create a new column 'parental_education_numeric' that converts education levels into numerical values: 'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3
        """
        # Calculate total leisure hours
        self.df['total_leisure_hours'] = self.df['social_media_hours'] + self.df['netflix_hours']
        
        # Calculate study efficiency, handling division by zero
        self.df['study_efficiency'] = self.df['exam_score'] / self.df['study_hours_per_day']
        self.df.loc[self.df['study_hours_per_day'] == 0, 'study_efficiency'] = 0
        
        # Create age groups using loc to avoid chained assignment
        self.df['age_group'] = '<18'  # Default value
        self.df.loc[self.df['age'] >= 22, 'age_group'] = '22+'
        self.df.loc[(self.df['age'] >= 18) & (self.df['age'] < 22), 'age_group'] = '18-21'
        
        # Map parental education levels to numeric values
        education_map = {'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3}
        self.df['parental_education_numeric'] = self.df['parental_education_level'].map(education_map)

    """ Analysis Functions
    - Returns descriptive statistics of the dataset
    """
    def get_summary(self):
        return self.df.describe()

    """ Analysis Functions
    - Identifies and returns the count of missing values per column
    """
    def get_missing_values(self):
        return self.df.isnull().sum()

    """ Analysis Functions
    - Return the mean of the exam scores
    """
    def get_score_mean(self):
        return self.df['exam_score'].mean()

    """ Analysis Functions
    - Computes average (mean) exam scores based on grouping by a given column
    """
    def score_by_column(self, column):
        return self.df.groupby(column)['exam_score'].mean()

    """ Analysis Functions
    - Returns the average (mean) exam score for each unique number of study hours
    """ 
    def score_by_study_hours(self):
        return self.df.groupby('study_hours_per_day')['exam_score'].mean() 