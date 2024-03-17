# Mushroom Classification Project

This project aims to classify mushrooms into edible and poisonous categories based on various features. It utilizes machine learning techniques to build predictive models and visualize the data.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Visualizations](#visualizations)
7. [Usage](#usage)
8. [Requirements](#requirements)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

The Mushroom Binary Classification Project focuses on predicting whether mushrooms are edible or poisonous based on features such as cap shape, odor, gill color, etc. Accurate classification is vital for mushroom enthusiasts and farmers to distinguish safe mushrooms from toxic ones.

## Dataset

The dataset used in this project is the Mushroom Classification dataset obtained from Kaggle. It contains various attributes of mushrooms, including categorical features that require one-hot encoding before model training.

## Preprocessing

Preprocessing steps include one-hot encoding of categorical features, splitting the dataset into training and testing sets.

## Model Training

The project utilizes Support Vector Machines (SVM) for classification. Hyperparameter tuning is performed using GridSearchCV to find the best combination of parameters such as C, kernel type, and gamma value.

## Evaluation

Model performance is evaluated using accuracy, confusion matrix. Visualizations help understand the distribution of data and decision boundaries.

## Visualizations

Visualizations include PCA visualization of the dataset and confusion matrix heatmap to illustrate model performance.

## Usage

To use the project, clone the repository and ensure the required dependencies are installed. Run the main script to preprocess the data, train the model, and evaluate its performance.

## Requirements

- Python 3
- scikit-learn
- pandas
- matplotlib
- seaborn

For detailed installation instructions, refer to the `requirements.txt` file.

## Contributing

Contributions to the project are welcome! Feel free to submit bug reports, feature requests, or pull requests via GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for details.