# Face Recognition Project – Team Setup Guide

This repository contains a Python-based face recognition project.
This README explains how team members can clone the repository, set up the environment, and run the project locally.

----------------------------------------------------------------

CLONE REPOSITORY

Clone the repository from GitHub and enter the project directory:

git clone https://github.com/USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME

Replace USERNAME/REPOSITORY_NAME with the actual GitHub repository URL.

----------------------------------------------------------------

CREATE VIRTUAL ENVIRONMENT

Creating a virtual environment is strongly recommended.

python3 -m venv venv

Activate the virtual environment:

Linux / macOS:
source venv/bin/activate

Windows:
venv\Scripts\activate

If activated correctly, (venv) will appear in the terminal.

----------------------------------------------------------------

INSTALL DEPENDENCIES

Install all required Python packages:

pip install -r requirements.txt

----------------------------------------------------------------

PROJECT STRUCTURE

.
├── data/
│   └── train/            Training images (one folder per person)
├── src/                  Source code
├── venv/                 Virtual environment (local only, not pushed to GitHub)
├── app.py                Main Streamlit application
├── requirements.txt      Python dependencies
├── README.md
└── .gitignore

The venv folder is intentionally excluded from the repository and must be created locally.

----------------------------------------------------------------

DATASET SETUP

Prepare the dataset using the following structure:

data/train/
├── PersonA/
│   ├── image1.jpg
│   └── image2.jpg
├── PersonB/
│   ├── image1.jpg
│   └── image2.jpg

Each folder name represents the identity label.

----------------------------------------------------------------

RUN APPLICATION

Make sure the virtual environment is active, then run:

streamlit run app.py

The application will open automatically in your default web browser.

----------------------------------------------------------------

TEAM GIT WORKFLOW

Before making any changes, always pull the latest version:

git pull origin main

After making changes:

git add .
git commit -m "Describe your changes briefly"
git push origin main

----------------------------------------------------------------

FILES THAT MUST NOT BE PUSHED

The following files and folders should never be pushed to GitHub:

- venv/
- __pycache__/
- .env

These files are already listed in .gitignore.

----------------------------------------------------------------

COMMON ISSUES

ModuleNotFoundError:
- Make sure the virtual environment is activated
- Make sure dependencies are installed using pip install -r requirements.txt

Streamlit command not found:
pip install streamlit

----------------------------------------------------------------

NOTES

- This project does not use deep learning models.
- External libraries are used only for image handling or face localization.
- All vector operations and distance metrics are implemented manually.
