# Text Detoxification Project

## Overview

The Text Detoxification project focuses on the development of a machine learning solution to detoxify toxic text, ensuring that it is transformed into non-toxic content while preserving its original context and meaning. This README file provides an overview of the project's structure, contents, and objectives.

**Group Number:** BS-21-AI

**Author:** Yazan Kbaili
**Email:** y.kbaili@innopolis.university

## Project Structure

The project directory is structured as follows:

- **data**: This directory contains subdirectories for managing data:
  - **raw**: Stores the original, immutable data used in the project.

- **notebooks**: This directory contains Jupyter notebooks used for various stages of the project. Notebooks follow a naming convention for easy ordering and identification.

- **reports**: Contains generated analysis and reports in various formats such as HTML, PDF, and LaTeX.

- **requirements.txt**: This file specifies the project's dependencies and is used to reproduce the analysis environment.

- **src**: The source code directory contains subdirectories for different project components:
  - **data**: Includes scripts for data downloading and preprocessing.
  - **models**: Contains scripts for training and making predictions using machine learning models.
  - **visualization**: Includes scripts for creating exploratory and results-oriented visualizations.

## Usage

To get started with the project, follow these steps:

1. Clone the repository to your local machine:

   git clone https://github.com/your-username/text-detoxification.git

2. Set up the Python environment by installing the required dependencies:

    pip install -r requirements.txt

3. Transform Data: To transform and preprocess the data, open a new notebook and the functions in the package data

4. Train Models: Refer to the model training notebooks to train machine learning models.

5. Make Predictions: Use the prediction funcion located in  src/models
