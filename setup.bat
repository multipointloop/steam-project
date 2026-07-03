@echo off
echo ============================================================
echo   Steam Game Analytics - Setup
echo ============================================================
echo.
echo Step 1/4: Installing Python dependencies...
pip install -q kagglehub pandas numpy matplotlib seaborn scikit-learn flask
echo.

echo Step 2/4: Downloading dataset from Kaggle...
python data_acquisition.py
echo.

echo Step 3/4: Cleaning data...
python data_cleaning.py
echo.

echo Step 4/4: Training model and building database...
python train_model.py
echo.

echo ============================================================
echo   Setup complete! Run: python app.py
echo   Then open: http://localhost:5000
echo ============================================================
pause
