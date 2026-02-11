❤️ Heart Disease Prediction Tool
This project features a Machine Learning application designed to identify potential heart disease cases with high sensitivity. The core objective is to maximize Recall, ensuring that as few sick patients as possible go undetected (minimizing False Negatives).
The application is deployed using Streamlit and uses a Neural Network (MLP) trained on clinical data.

🚀 Live Demo
[Link to your Streamlit App if deployed]

🧠 Model Strategy: Why Recall?
In medical diagnostics, missing a sick patient (False Negative) is much more dangerous than a false alarm (False Positive). Therefore, this model was tuned to prioritize:
Recall (Sensitivity): High priority to catch all positive cases.
Feature Selection: Reduced from 15 to 7 key clinical indicators using Permutation Importance.

🛠️ Tech Stack
Python 3.x
Scikit-Learn: MLPClassifier (Neural Networks).
Pandas & Numpy: Data manipulation.
Matplotlib & Seaborn: Data visualization.
Streamlit: Frontend web application.
Joblib: Model persistence.

📊 Key Features Used
The model uses the 7 most impactful features found in the Kaggle Heart Failure Dataset:
ST_Slope_Flat & ST_Slope_Up: ST segment morphology.
Cholesterol: Serum cholesterol levels.
Oldpeak: ST depression induced by exercise.
ChestPainType (ATA/NAP): Clinical pain assessment.
FastingBS: Blood sugar levels.
