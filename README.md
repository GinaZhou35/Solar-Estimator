# Solar-Estimator

Calgary solar energy and power estimator

BrainStation Data Science Bootcamp Capstone Project
Gina Zhou

Janurary 2022 Maple Cohort


Technical Summary
Refer to Capstone Final Report.pdf for a brief technical summary of the project.

gina_capstone_env.txt file that lists all the required packages for the project environment (running all the Jupyter notebooks and app).

Jupyter Notebooks
Files related to Jupyter notebooks:

"notebooks" folder which contains the following:
Final project-1 EDA.ipynb
Final project-2 Power prediction modeling and evaluation.ipynb
Final project-3 Energy prediction modeling and evaluation.ipynb


The original dataset files can be downloaded from 
 https://maps.nrel.gov/nsrdb-viewer/?aL=x8CI3i%255Bv%255D%3Dt%26Jea8x6%255Bv%255D%3Dt%26Jea8x6%255Bd%255D%3D1%26VRLt_G%255Bv%255D%3Dt%26VRLt_G%255Bd%255D%3D2%26mcQtmw%255Bv%255D%3Dt%26mcQtmw%255Bd%255D%3D3&bL=clight&cE=0&lR=0&mC=4.740675384778373%2C22.8515625&zL=2


Web Application
Due to the limitation on the file size, the model that was implemented in the app is not in cluded in the folder.
Files and information related to Web_app folder:
app.py – streamlit app script
app_energy.zip. – monthly GHI data generated by SARIMA model in energy prediction Jupyter notebook
solar.png – graph used in the app
live weather data shown in app is from openweatherapp API
python pysolar library was installed to get the solar zenith angle data.

To run the app on your local machine, type streamlit run app.py into command line from the repo directory.

For questions, please feel free to contact me at wenjin.zhou35@gmail.com.
