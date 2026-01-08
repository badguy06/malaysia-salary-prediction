import pandas as pd
import joblib
import numpy as np

# Load tuned model
best_model = joblib.load('salary_prediction_model_tuned.pkl')

def experience_bucket(x):
    if x < 1:
        return "Intern/<1"
    elif x <= 3:
        return "1-3"
    elif x <= 5:
        return "4-5"
    elif x <= 10:
        return "6-10"
    else:
        return "10+"




def predict_salary(job_title, category, experience, state):
    exp_bucket = experience_bucket(experience)
    sample = pd.DataFrame([{
        'job_title_normalized': job_title,
        'category_clean': category,
        "experience_bucket": exp_bucket,
        'state_region': state
    }])
    
    log_salary_pred = best_model.predict(sample)[0]
    salary_myr = np.expm1(log_salary_pred)
    return round(salary_myr, 2)

def predict_salary_range(job_title, category, experience, state, pct=0.15):
    salary = predict_salary(job_title, category, experience, state)
    low = salary * (1 - pct)
    high = salary * (1 + pct)
    return round(low, 2), round(high, 2)