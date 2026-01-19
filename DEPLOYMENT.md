# Deployment Guide for Render.com

## Files Created for Deployment

1. **runtime.txt** - Specifies Python 3.12.10 (to avoid Python 3.13 compatibility issues)
2. **Procfile** - Tells Render how to run the application
3. **render.yaml** - Render configuration file
4. **requirements.txt** - Updated with compatible package versions

## Deployment Steps

1. **Push your code to GitHub**

2. **Connect to Render:**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure the service:**
   - **Name:** phishing-url-detector
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `gunicorn app:app`
   - **Python Version:** Will use Python 3.12.10 from runtime.txt

4. **Deploy:**
   - Click "Create Web Service"
   - Render will build and deploy your application

## Important Notes

- The model will be trained during the build process
- The application uses Gunicorn for production
- Python 3.12.10 is used to avoid compatibility issues with older packages
- All dependencies are specified in requirements.txt

## Troubleshooting

If you encounter build errors:
1. Check that `phishing.csv` is in your repository
2. Verify all files are committed to Git
3. Check Render logs for specific error messages
