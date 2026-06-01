# CTR-Prediction-Project
🚀 Project Overview
This project develops a high-performance machine learning model to predict the Click-Through Rate (CTR) of mobile ads. The ability to accurately predict whether a user will click on an ad is a multi-billion dollar problem and a core task for any company operating in the digital advertising space.

This project tackles this challenge by building and evaluating a sophisticated classification model on a large-scale, real-world dataset. The final model serves as a powerful tool for optimizing ad placements and understanding the key drivers of user engagement.

Basic Live Demo: https://ctr-prediction-project-ykm24ryy8xktlbrgkhu5bz.streamlit.app/

Key Features
Handles a large-scale dataset by employing a practical sampling strategy.
Performs time-based feature engineering to extract predictive signals from timestamps.
Trains a high-performance LightGBM classifier, an industry-standard for tabular data.
Achieves a strong LogLoss score, demonstrating high accuracy on an imbalanced dataset.
Includes an interactive UI built with ipywidgets for real-time model exploration.
📊 Dataset
The model was trained on the Avazu CTR Prediction dataset, a well-known benchmark from a Kaggle competition. The dataset contains 10 days of click-through data and is comprised of 23 anonymized categorical features.

Dataset Link: Click-Through Rate (CTR) Prediction Dataset on Kaggle
🛠️ Methodology & Workflow
The project followed a complete, end-to-end machine learning workflow:

Data Loading & Sampling: The original dataset is massive (>40 million rows). To handle this on a standard machine, a 2-million-row sample was loaded for analysis and model training. This is a key practical skill for handling big data.

Exploratory Data Analysis (EDA): The initial analysis revealed that the dataset is composed almost entirely of categorical features, many of which have a very high number of unique values (high cardinality). Crucially, the target variable click was found to be highly imbalanced, with clicks (1) being much rarer than non-clicks (0).

Feature Engineering: To improve the model's performance, the hour timestamp column was converted into more useful features: day_of_week and hour_of_day. This allows the model to learn time-based patterns in user behavior.

Model Training: A LightGBM Classifier was selected for its speed, memory efficiency, and excellent performance on large, categorical datasets. The data was split into an 80% training set and a 20% testing set for robust evaluation.

Evaluation: The model was evaluated using the LogLoss metric. This metric is ideal for classification problems where the goal is to predict probabilities, as it heavily penalizes models that are overconfident in wrong predictions.

✅ Results & Performance
The trained LightGBM model achieved a final LogLoss score of 0.3752 on the unseen test set. This strong score indicates that the model is highly effective at predicting the probability of an ad being clicked, performing significantly better than a random guess.

💡 Crucial Learnings
This project provided several key learnings relevant to real-world data science roles:

Handling Big Data: Gained practical experience in using sampling techniques to explore and model datasets that do not fit into memory.
The Power of Feature Engineering: Demonstrated how creating simple, intuitive features (like hour_of_day) from existing data can provide powerful predictive signals.
Choosing the Right Tool: Understood why advanced, efficient models like LightGBM are the industry standard over models like Random Forest for large-scale, high-cardinality tabular datasets.
Working with Imbalanced Data: Gained experience with a classic imbalanced classification problem and the importance of using appropriate evaluation metrics like LogLoss instead of simple accuracy.
🎛️ Interactive Demo (UI)
An interactive UI was built directly in the notebook using ipywidgets to demonstrate the model's functionality. The tool allows a user to:

Load a random ad from the test set.
Modify key features like the hour of the day or day of the week.
Receive a new, real-time Predicted Click-Through Rate from the model based on these changes.
This provides an intuitive way to explore the model and understand how different factors influence its predictions.

💻 Technology Stack
Language: Python
Libraries: Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib, Seaborn, ipywidgets
⚙️ How to Run
Clone the repository.
Install dependencies from the requirements.txt file.
Download the dataset from the link provided above.
Run the notebook: The main analysis and model training process is contained in the .ipynb notebook file. Ensure the file path to the dataset is updated.
