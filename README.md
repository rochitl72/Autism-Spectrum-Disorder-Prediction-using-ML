# 🧠 Autism Spectrum Disorder Prediction App

A machine learning-based web app that predicts whether a person is likely to exhibit traits of Autism Spectrum Disorder (ASD) based on input screening questions and demographics. Built with Scikit-learn, Streamlit, and real-world ASD screening dataset.

---

## 🎯 Features

- 🔍 Real-time ASD prediction from 10-question input form
- 📈 Trained Logistic Regression model with 100% accuracy on test set
- 📦 Joblib-exported model, scaler, and encoders
- 🖥️ Streamlit UI with Dark Mode and Clean UX
- 🗃️ Prediction logs saved with timestamp to CSV

---

## 💻 Tech Stack

| Part          | Library/Tech           |
|---------------|------------------------|
| ML Model      | `scikit-learn`         |
| UI            | `streamlit`            |
| Data Handling | `pandas`, `numpy`      |
| Export        | `joblib`               |

---

## 🗂️ Folder Structure

autism_predictor/ ├── backend/ # ML training and model exporting │ └── train_model.py ├── data/ # autism_data.csv dataset ├── frontend/ # Streamlit app │ └── app.py ├── model/ # .joblib model files ├── logs/ # Prediction logs ├── README.md # You are here! ├── requirements.txt └── .gitignore

yaml
Copy
Edit

---

## 🛠️ How to Run Locally

### 1. Clone & Set up Environment

```bash
git clone https://github.com/YOUR_USERNAME/Autism-Predictor-App.git
cd Autism-Predictor-App
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2. Train the Model (optional)
bash
Copy
Edit
python backend/train_model.py
3. Launch the Web App
bash
Copy
Edit
cd frontend
streamlit run app.py
