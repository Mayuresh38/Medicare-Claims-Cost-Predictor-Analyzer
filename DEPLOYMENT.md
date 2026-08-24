# Deployment Guide: Medicare Claims Cost Predictor & Analyzer

This Streamlit application can be deployed to the cloud for free using either **Streamlit Community Cloud** or **Hugging Face Spaces**. 

---

## Option 1: Streamlit Community Cloud (Recommended)

Streamlit Community Cloud connects directly to your GitHub repository and automatically deploys your app whenever you push updates.

### Steps to Deploy:
1. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in (or sign up) using your GitHub account.
2. Click **New app** (or **Create app**) in the top right corner.
3. Fill in the following repository details:
   - **Repository**: `Mayuresh38/Medicare-Claims-Cost-Predictor-Analyzer`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**

Your app will build, install dependencies listed in `requirements.txt`, and go live in a few minutes.

---

## Option 2: Hugging Face Spaces

Hugging Face Spaces offers free hosting for machine learning and data science web applications (including Streamlit).

### Steps to Deploy:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and sign in.
2. Click **Create new Space**.
3. Configure your Space:
   - **Space name**: Choose a name (e.g., `medicare-cost-predictor`).
   - **SDK**: Select **Streamlit**.
   - **Space Hardware**: Choose the default free tier (**CPU basic**).
   - **Visibility**: Public or Private.
4. Click **Create Space**.
5. Once created, you can connect it directly to your GitHub repository or push your files directly to the Hugging Face Git remote:
   ```bash
   # Add Hugging Face as a remote
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   # Push your code
   git push -f hf main
   ```

---

## Option 3: Local Deployment (Run Locally)

To run the application locally on your machine:
1. Clone the repository and navigate into the folder:
   ```bash
   git clone https://github.com/Mayuresh38/Medicare-Claims-Cost-Predictor-Analyzer.git
   cd Medicare-Claims-Cost-Predictor-Analyzer
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
