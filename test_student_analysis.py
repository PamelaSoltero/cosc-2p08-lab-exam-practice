import pytest
import pandas as pd
import numpy as np
from student_analysis import StudentAnalysis
#from solution.student_analysis_solution import StudentAnalysis

@pytest.fixture
def student_analyzer():
    filepath = './student_habits_performance.csv'
    return StudentAnalysis(filepath)

@pytest.fixture
def processed_analyzer(student_analyzer):
    student_analyzer.clean_data()
    student_analyzer.transform_data()
    return student_analyzer

def test_clean_data(student_analyzer):
    """Test data cleaning steps."""
    student_analyzer.clean_data()
    
    # Check if student_id is dropped
    assert 'student_id' not in student_analyzer.df.columns
    
    # Check if numerical columns have no missing values
    numerical_cols = ['age', 'exam_score', 'study_hours_per_day', 'social_media_hours', 'netflix_hours']
    for col in numerical_cols:
        if col in student_analyzer.df.columns:
            assert student_analyzer.df[col].isnull().sum() == 0
    
    # Check if categorical column has no missing values
    if 'parental_education_level' in student_analyzer.df.columns:
        assert student_analyzer.df['parental_education_level'].isnull().sum() == 0
    
    # Check if Yes/No columns are converted to 1/0
    yes_no_cols = ['part_time_job', 'extracurricular_participation']
    for col in yes_no_cols:
        if col in student_analyzer.df.columns:
            assert set(student_analyzer.df[col].unique()).issubset({0, 1})

def test_transform_data(processed_analyzer):
    """Test data transformation steps."""
    # Check if new columns exist
    assert 'total_leisure_hours' in processed_analyzer.df.columns
    assert 'study_efficiency' in processed_analyzer.df.columns
    assert 'age_group' in processed_analyzer.df.columns
    assert 'parental_education_numeric' in processed_analyzer.df.columns
    
    # Check total_leisure_hours calculation
    assert (processed_analyzer.df['total_leisure_hours'] == 
            processed_analyzer.df['social_media_hours'] + processed_analyzer.df['netflix_hours']).all()
    
    # Check study_efficiency calculation and handling of zero study hours
    assert not np.isinf(processed_analyzer.df['study_efficiency']).any()
    if (processed_analyzer.df['study_hours_per_day'] == 0).any():
        assert (processed_analyzer.df.loc[processed_analyzer.df['study_hours_per_day'] == 0, 'study_efficiency'] == 0).all()
    
    # Check age_group categories
    age_groups = processed_analyzer.df['age_group'].unique()
    assert all(group in ['<18', '18-21', '22+'] for group in age_groups)
    
    # Check parental_education_numeric mapping
    education_map = {'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3}
    for education_level, numeric_value in education_map.items():
        mask = processed_analyzer.df['parental_education_level'] == education_level
        if mask.any():
            assert (processed_analyzer.df.loc[mask, 'parental_education_numeric'] == numeric_value).all()

def test_get_summary(processed_analyzer):
    """Test the summary statistics function."""
    summary = processed_analyzer.get_summary()
    assert isinstance(summary, pd.DataFrame)
    assert 'count' in summary.index
    assert 'mean' in summary.index
    assert 'std' in summary.index

def test_get_missing_values(student_analyzer):
    """Test the missing values check."""
    missing_values = student_analyzer.get_missing_values()
    assert isinstance(missing_values, pd.Series)
    assert pd.api.types.is_integer_dtype(missing_values.dtype)

def test_get_score_mean(processed_analyzer):
    """Test the calculation of the mean exam score."""
    mean_score = processed_analyzer.get_score_mean()
    assert isinstance(mean_score, (float, np.floating))
    assert 0 <= mean_score <= 100

def test_score_by_column(processed_analyzer):
    """Test grouping scores by a column."""
    # Test with gender column if it exists
    if 'gender' in processed_analyzer.df.columns:
        scores_by_gender = processed_analyzer.score_by_column('gender')
        assert isinstance(scores_by_gender, pd.Series)
        assert pd.api.types.is_numeric_dtype(scores_by_gender.dtype)

def test_score_by_study_hours(processed_analyzer):
    """Test grouping scores by study hours."""
    scores_by_hours = processed_analyzer.score_by_study_hours()
    assert isinstance(scores_by_hours, pd.Series)
    assert pd.api.types.is_numeric_dtype(scores_by_hours.dtype)
    assert len(scores_by_hours) > 0