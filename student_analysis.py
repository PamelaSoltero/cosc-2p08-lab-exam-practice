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
        ## ADD YOUR CODE HERE
        pass # Remove this pass when you add code

    """ Data Transformation
    - Create a new column 'total_leisure_hours' that combines time spent on social media and Netflix
    - Create a new column 'study_efficiency' that measures how well study time translates to exam performance (set to 0 if study hours is 0)
    - Create a new column 'age_group' that categorizes students into three age ranges: '<18', '18-21', '22+'
    - Create a new column 'parental_education_numeric' that converts education levels into numerical values: 'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3
    """
    def transform_data(self):
        ## ADD YOUR CODE HERE
        pass # Remove this pass when you add code

    """ Analysis Functions
    - Returns descriptive statistics of the dataset
    """
    def get_summary(self):
        return ## ADD YOUR CODE HERE

    """ Analysis Functions
    - Identifies and returns the count of missing values per column
    """
    def get_missing_values(self):
        return ## ADD YOUR CODE HERE

    """ Analysis Functions
    - Return the mean of the exam scores
    """
    def get_score_mean(self):
        return ## ADD YOUR CODE HERE

    """ Analysis Functions
    - Computes average (mean) exam scores based on grouping by a given column
    """
    def score_by_column(self, column):
        return ## ADD YOUR CODE HERE

    """ Analysis Functions
    - Returns the average (mean) exam score for each unique number of study hours
    """
    def score_by_study_hours(self):
        return ## ADD YOUR CODE HERE