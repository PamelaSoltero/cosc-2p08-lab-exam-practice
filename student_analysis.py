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
        self.df.drop(['student_id'], axis =1, inplace = True)
        
        num_cols = ['age', 'exam_score', 'study_hours_per_day', 'social_media_hours', 'netflix_hours']
        for col in num_cols:
            median_value = self.df[col].median()
            self.df[col].fillna(median_value, inplace=True)
        
        most_common_edu = self.df['parental_education_level'].mode()[0]
        self.df['parental_education_level'].fillna(most_common_edu, inplace= True)

        self.df['part_time_job']= self.df['part_time_job'].map({"Yes" : 1, "No":0})
        self.df['extracurricular_participation'] = self.df['extracurricular_participation'].map({"Yes" : 1, "No":0})
        

    """ Data Transformation
    - Create a new column 'total_leisure_hours' that combines time spent on social media and Netflix
    - Create a new column 'study_efficiency' that measures how well study time translates to exam performance (set to 0 if study hours is 0)
    - Create a new column 'age_group' that categorizes students into three age ranges: '<18', '18-21', '22+'
    - Create a new column 'parental_education_numeric' that converts education levels into numerical values: 'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3
    """
    def transform_data(self):
        self.df['total_leisure_hours'] = self.df['social_media_hours'] + self.df['netflix_hours']
        self.df['study_efficiency'] = self.df['study_hours_per_day'] / self.df['exam_score']
        self.df['exam_score'][self.df['study_hours_per_day'] == 0 ] = 0
        self.df['age_group'] = "<18"
        self.df['age_group'][self.df['age'] >= 18] = "18-21"
        self.df['age_group'][self.df['age'] >= 22] = "22+"
        self.df['parental_education_numeric'] = self.df['parental_education_level'].map({'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3})
        pass # Remove this pass when you add code

    """ Analysis Functions
    - Returns descriptive statistics of the dataset
    """
    def get_summary(self):
        return self.df.describe()

    """ Analysis Functions
    - Identifies and returns the count of missing values per column
    """
    def get_missing_values(self):
        return self.df.isna().sum()

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

    """ Analysis Functions
    - Returns the count of each unique value in the age column
    """
    def get_age_value_counts(self):
        return self.df['age'].value_counts()

    """ Analysis Functions
    - Returns the count of each unique value in the study_hours_per_day column
    """
    def get_study_hours_value_counts(self):
        return self.df['study_hours_per_day'].value_counts()

    """ Analysis Functions
    - Returns the minimum exam score
    """
    def get_min_score(self):
        return self.df['exam_score'].min()

    """ Analysis Functions
    - Returns the maximum exam score
    """
    def get_max_score(self):
        return self.df['exam_score'].max()

    """ Analysis Functions
    - Returns the sum of all study hours
    """
    def get_total_study_hours(self):
        return self.df['study_hours_per_day'].sum()

    """ Analysis Functions
    - Returns the sum of all social media hours
    """
    def get_total_social_media_hours(self):
        return self.df['social_media_hours'].sum()
