# Disaster Response Pipeline Project
This project classifies emergency messages in categories like (related, water, ...)
### Installation:
1. The python scripts has been tested with version 3.6.3 and 3.8.1
2. Use 'pip install' for prerequisite packages pandas, plotly, nltk, sklearn, sqlalchemy, and flask
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
### Files:
```bash
.
├── app
│   ├── run.py: for running flask and deploying ml pipeline in backend
│   └── templates
│       ├── go.html:
│       └── master.html: main page
├── data
│   ├── disaster_categories.csv: Input categories classified for messages
│   ├── disaster_messages.csv: Input messages
│   ├── DisasterResponse.db: Output database from script 'process_data.py'
│   └── process_data.py: Python script for cleaning and one-hot encoding of input dataset
├── models
│   ├── classifier.pkl: pickle model of machine learning pipeling
│   └── train_classifier.py: python script for building, training, optimizing, and testing machine learning pipeline for predicting categories(36) of input messages.
└── README.md
```