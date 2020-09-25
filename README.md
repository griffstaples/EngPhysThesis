# Design of a New Blood Pressure Monitor

### Overview

This paper and corresponding neural network were created for my 4th Year Engineering Physics Thesis - an 8 month research and design project.

calculate_blood_pressure.py takes in data created by create_data.py and data_processing and trains a neural network to predict blood pressure based off of a public PPG dataset. Either systolic or diastolic blood pressure can be calculated by simply commenting and uncommenting line 23 and 24 of calculate_blood_pressure.py. This method of prediction was determined to not provided sufficiently accurate results. However, a key blood pressure correlate, Pulse Transit Time (PTT), was not present in the data set used therefore accuracy would likely greatly improve with such information.

To learn more about the proposed blood pressure monitor, please read "Design of a New Blood Pressure Monitor.pdf" provided in this repository.