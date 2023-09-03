# Spaceship Passenger Transport Prediction

## Overview

This repository contains a machine learning project aimed at predicting whether passengers on the Spaceship would be transported to their destination in another dimension. The project utilizes a dataset consisting of personal records of passengers, with 13 variables including passenger information, cabin details, amenities usage, and more.

The primary goal of this project is to develop a predictive model using XGBoost, a powerful gradient boosting algorithm, to determine whether a passenger would be transported to another dimension during the voyage. The model was trained on a substantial portion of the dataset, and its performance has been evaluated using systematic data preprocessing, hyperparameter tuning, and cross-validation techniques.

## Dataset

- **train.csv**: This file contains personal records for approximately two-thirds of the Spaceship Titanic passengers. It includes information such as PassengerId, HomePlanet, CryoSleep status, Cabin details, Destination, Age, VIP status, and the amount spent on various amenities.

## Project Structure

The project is organized as follows:

- **data**: This directory contains the dataset file, "train.csv."

- **notebooks**: This directory contains Jupyter notebooks used for data exploration, preprocessing, model training, and evaluation.

- **models**: The trained XGBoost model is saved here for future use.

- **presentation slides**:

We have prepared a set of PowerPoint (PPT) slides for our group presentation on this project, which was delivered during the Data Science Programming class in summer 2023. You can access the presentation slides using the following link: - [Presentation Slides (PPT)]((https://github.com/lisahyuniko/spaceship-prediction/blob/main/ML_Spaceship_Titanic_Project%20Slides.pdf))

Feel free to review the slides for a more detailed overview of our project and findings.


- **README.md**: The main documentation file providing an overview of the project.

## Model Performance

The selected XGBoost model has demonstrated high accuracy on the test set, indicating its potential for real-world predictions. The model's performance and feature importance have been visualized to aid in interpretation and understanding of the predictive results.

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine.

2. Install the required Python libraries listed in the "requirements.txt" file.

3. Navigate to the "notebooks" directory and open the Jupyter notebooks to explore the data, preprocess it, train the model, and evaluate its performance.

4. The trained model can be found in the "models" directory for future predictions or deployment.

## Dependencies

This project relies on several Python libraries, which are listed in the "requirements.txt" file. You can install them using pip:

```bash
pip install -r requirements.txt
