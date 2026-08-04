"""
constants.py
------------
Shared display labels and generic educational guidance, used by both the
Streamlit app (app.py) and the REST API (api.py) so they stay in sync.
"""

DATASET_LABELS = {
    "breast_cancer": "Breast Cancer",
    "heart": "Heart Disease",
    "diabetes": "Diabetes",
}

DISEASE_LABELS = {
    "breast_cancer": "Malignant Tumor",
    "heart": "Heart Disease",
    "diabetes": "Diabetes",
}

# Simple, generic educational guidance per dataset — not medical advice.
DOCTOR_MAP = {
    "breast_cancer": "Oncologist",
    "heart": "Cardiologist",
    "diabetes": "Endocrinologist / General Physician",
}

RECOMMENDATIONS = {
    "breast_cancer": [
        "Share this result with an oncologist for confirmatory imaging/biopsy.",
        "Don't self-diagnose from this alone — it's a screening aid, not a diagnosis.",
        "Keep a record of any changes noticed and when they started.",
    ],
    "heart": [
        "Monitor blood pressure and resting heart rate regularly.",
        "Reduce sodium and saturated fat intake; stay physically active if cleared to do so.",
        "Seek urgent care immediately for chest pain, shortness of breath, or fainting.",
    ],
    "diabetes": [
        "Get a fasting blood glucose / HbA1c test to confirm.",
        "Stay hydrated, moderate sugar/refined-carb intake, and keep up regular activity.",
        "Watch for excessive thirst, frequent urination, or unexplained fatigue.",
    ],
}
