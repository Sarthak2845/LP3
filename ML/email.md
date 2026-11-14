# **Email Spam Detection Using Binary Classification: KNN and SVM**

Let's understand this assignment from the absolute basics so that you can explain it in your practical examination, write the theory in your journal, and understand how the algorithms actually work.

**1. Introduction to Email Spam Detection**

Email Spam Detection is a machine learning application used to identify whether an incoming email is genuine or unwanted.

Spam detection is a **Binary Classification** problem because the output is divided into exactly two classes.

For example:

- Normal email: An email containing genuine or useful information.
- Spam email: An unwanted email that may contain advertisements, fraudulent links, phishing attempts, or misleading information.

**2. What is Binary Classification?**

Binary Classification is a supervised machine learning technique used to classify data into one of two possible categories.

In this assignment, we have two states:

| State          | Classification | Meaning        |
| -------------- | -------------- | -------------- |
| Normal State   | Not Spam (0)   | Genuine email  |
| Abnormal State | Spam (1)       | Unwanted email |

**How does it work?**

**Input: Incoming Email**

Example: "Congratulations! You won a free iPhone."

**Feature Extraction**

Words, links, sender information, keywords, etc.

**Machine Learning Classifier**

KNN or SVM algorithm

**Not Spam (0)**

Genuine email

**Spam (1)**

Unwanted email

**3. What is Machine Learning?**

Machine Learning is a branch of Artificial Intelligence (AI) that enables computers to learn from data and make predictions without being explicitly programmed for every situation.

In spam detection, the machine learning model is trained using previously classified emails.

For example:

| Email                               | Label    |
| ----------------------------------- | -------- |
| Meeting scheduled for tomorrow      | Not Spam |
| Win a free lottery prize            | Spam     |
| Please find the attached assignment | Not Spam |
| Claim your free cash reward         | Spam     |

The model learns patterns from these examples and uses them to classify new emails.

**Types of Machine Learning**

1. Supervised Learning: The model learns from labelled data.
2. Unsupervised Learning: The model identifies patterns in unlabelled data.
3. Reinforcement Learning: The model learns through rewards and penalties.

Our assignment uses **Supervised Learning**, because the training emails already have labels (Spam or Not Spam).

**4. Feature Extraction in Email Spam Detection**

Before applying machine learning algorithms, we need to convert email text into numerical data.

Why?

Because machine learning algorithms such as KNN and SVM work with numerical features rather than directly understanding natural language.

Some important features are:

| Feature            | Description                        |
| ------------------ | ---------------------------------- |
| Number of words    | Total words in the email           |
| Spam keywords      | Words like FREE, WIN, OFFER        |
| Number of links    | Number of URLs present             |
| Special characters | Number of !, $, %, etc.            |
| Sender information | Information about the email sender |
| Word frequency     | Frequency of particular words      |

**Example:**

Email: "Congratulations! You won a FREE prize. Click here!"

Extracted features might look like:

| Feature           | Value |
| ----------------- | ----- |
| Number of words   | 8     |
| Spam keywords     | 2     |
| Number of links   | 0     |
| Exclamation marks | 2     |

These numerical values are given to the classification algorithms.

In real-world systems, text is commonly converted using techniques such as:

- Bag of Words (BoW)
- TF-IDF (Term Frequency–Inverse Document Frequency)

**5. K-Nearest Neighbors (KNN) Algorithm**

First classification algorithm

**What is KNN?**

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression.

It classifies a new data point based on the classes of its K nearest neighboring data points.

In simple words:

KNN checks the most similar emails in the training dataset and assigns the new email to the category that appears most frequently among its nearest neighbors.

**How does KNN work?**

Consider that we have already trained our model using several emails labelled as Spam and Not Spam.

Now, a new email arrives.

KNN performs the following steps:

1. Store the labelled training dataset.
2. Select the value of K (number of neighbors).
3. Calculate the distance between the new email and existing emails.
4. Identify the K nearest neighbors.
5. Count the classes of the nearest neighbors.
6. Assign the majority class as the final prediction.

**Example of KNN**

Suppose K = 5.

A new email is being classified. KNN finds its five nearest emails:

| Neighbor | Email Class |
| -------- | ----------- |
| 1        | Spam        |
| 2        | Spam        |
| 3        | Not Spam    |
| 4        | Spam        |
| 5        | Not Spam    |

Result:

- Spam votes = 3
- Not Spam votes = 2

Therefore, the new email is classified as **Spam**.

Class AClass BPredictionTie

k = 2: The 1–1 tie leaves no majority.

\\(k\\)

Give feedback

**Distance Formula**

KNN commonly uses Euclidean Distance to calculate the distance between two data points.

\\[ d=\sqrt{\sum\_{i=1}^{n}(x_i-y_i)^2} \\]

Where:

- \\(d\\) = Distance between two data points.
- \\(x_i\\) = Features of the first data point.
- \\(y_i\\) = Features of the second data point.
- \\(n\\) = Total number of features.

**What is the value of K?**

K represents the number of nearest neighbors considered during classification.

For example:

- K = 1: Considers only one nearest neighbor.
- K = 3: Considers three nearest neighbors.
- K = 5: Considers five nearest neighbors.

Choosing the correct K is important.

- Small K: May be affected by noise and overfitting.
- Large K: May produce underfitting and ignore local patterns.

An odd value of K is often selected in binary classification to reduce the possibility of ties.

**Advantages of KNN**

1. Simple and easy to understand.
2. Easy to implement.
3. No complex training process is required.
4. Can be used for classification and regression.
5. Works well with smaller datasets.

**Disadvantages of KNN**

1. Prediction can be slow for large datasets.
2. Requires more memory to store training data.
3. Sensitive to irrelevant features and noise.
4. Performance depends on the selection of K.
5. Feature scaling is important because distance calculations are affected by feature magnitudes.

**6. Support Vector Machine (SVM) Algorithm**

Second classification algorithm

**What is SVM?**

Support Vector Machine (SVM) is a supervised machine learning algorithm used mainly for classification problems.

It works by finding the best possible decision boundary (called a hyperplane) that separates data points belonging to different classes.

In simple words:

SVM tries to draw the best dividing line between Spam and Not Spam emails so that both categories are separated as accurately as possible.

**Important terms in SVM**

1. Hyperplane: A decision boundary that separates two classes.
2. Support Vectors: The data points closest to the decision boundary.
3. Margin: The distance between the hyperplane and the nearest data points of both classes.
4. Maximum Margin: SVM tries to maximize the distance between the two classes' nearest points and the decision boundary.

**How does SVM work?**

1. Take the labelled email dataset.
2. Convert email features into numerical values.
3. Identify the different classes.
4. Find a suitable hyperplane to separate the classes.
5. Select the hyperplane with the maximum margin.
6. Use the trained model to classify new emails.

**Example of SVM**

Suppose we have two categories:

- Spam emails
- Not Spam emails

SVM finds a decision boundary that separates these two categories.

[Support vector machines | Predictive Analytics in Business Class Notes](https://images.openai.com/static-rsc-4/4MFY7P8YsNBGW33KxC9z3D61g8UwMy659vgbmYsQ54lV3ejYPIKY5tvYtnDiSs24euKRPNs0XXsI6jCjeSx6zqpa5VVN-Y4pL7KRmTUCKNWeNILpqyphP2l83lbWnvNA8ptXYyoXLyWBOwes3P-RQNb4Hl_yOsAeJlCxzcePGUk?purpose=inline)

[fiveable.me](https://fiveable.me/predictive-analytics-in-business/unit-4/support-vector-machines/study-guide/1afMd7jSxizFGdnu?utm_source=chatgpt.com)

The line in the diagram represents the decision boundary. The closest points to the line are called support vectors.

A new email is classified according to which side of the decision boundary it falls on.

**Mathematical Formula**

The equation of a hyperplane is:

\\[ w^Tx+b=0 \\]

Where:

- \\(w\\) = Weight vector.
- \\(x\\) = Input feature vector.
- \\(b\\) = Bias.
- \\(0\\) = Decision boundary.

The classification decision can be represented as:

\\[ f(x)= \begin{cases} +1 & \text{Spam}\\\ -1 & \text{Not Spam} \end{cases} \\]

**Types of SVM**

1. Linear SVM: Used when data can be separated using a straight line or hyperplane.
2. Non-Linear SVM: Used when data cannot be separated using a straight line. It uses kernel functions to create more complex decision boundaries.

**Common Kernel Functions**

| Kernel     | Description                            |
| ---------- | -------------------------------------- |
| Linear     | Separates data using a linear boundary |
| Polynomial | Creates polynomial decision boundaries |
| RBF        | Handles complex non-linear patterns    |
| Sigmoid    | Uses a sigmoid-based kernel function   |

**Advantages of SVM**

1. Effective in high-dimensional datasets.
2. Works well for classification problems.
3. Can handle linear and non-linear data.
4. Uses support vectors to define the decision boundary.
5. Can provide good accuracy with properly selected parameters.

**Disadvantages of SVM**

1. Training can be slow for very large datasets.
2. Selecting the correct kernel and parameters can be difficult.
3. Requires feature scaling.
4. Less interpretable than simpler algorithms.
5. Performance may decrease when the dataset contains considerable noise or overlapping classes.

**7. Difference Between KNN and SVM**

This is an important comparison for your practical examination.

| KNN                                                 | SVM                                            |
| --------------------------------------------------- | ---------------------------------------------- |
| Instance-based learning algorithm.                  | Margin-based learning algorithm.               |
| Classifies using nearest neighbors.                 | Classifies using a hyperplane.                 |
| Uses distance calculations.                         | Uses support vectors and margins.              |
| No explicit model training is required.             | Requires model training.                       |
| Prediction can be slow for large datasets.          | Prediction is generally faster after training. |
| Sensitive to the value of K.                        | Sensitive to kernel and parameter selection.   |
| Suitable for smaller datasets and simpler problems. | Effective for high-dimensional classification. |

**8. Performance Analysis of KNN and SVM**

Performance analysis means evaluating how accurately and efficiently the algorithms classify emails into Spam and Not Spam categories.

We use different evaluation metrics to compare their performance.

**A. Confusion Matrix**

A Confusion Matrix is a table used to evaluate the performance of a classification model.

It compares actual values with predicted values.

| Actual / Predicted | Not Spam            | Spam                |
| ------------------ | ------------------- | ------------------- |
| Not Spam           | True Negative (TN)  | False Positive (FP) |
| Spam               | False Negative (FN) | True Positive (TP)  |

Understand these four terms:

- True Positive (TP): Spam email correctly identified as Spam.
- True Negative (TN): Normal email correctly identified as Not Spam.
- False Positive (FP): Normal email incorrectly identified as Spam.
- False Negative (FN): Spam email incorrectly identified as Not Spam.

**B. Evaluation Metrics**

**1. Accuracy**

Accuracy measures the percentage of correctly classified emails.

\\[ Accuracy=\frac{TP+TN}{TP+TN+FP+FN}\times100 \\]

**2. Precision**

Precision measures how many emails predicted as Spam were actually Spam.

\\[ Precision=\frac{TP}{TP+FP}\times100 \\]

**3. Recall**

Recall measures how many actual Spam emails were correctly identified.

\\[ Recall=\frac{TP}{TP+FN}\times100 \\]

**4. F1-Score**

F1-Score is the harmonic mean of Precision and Recall.

\\[ F1=2\times\frac{Precision\times Recall}{Precision+Recall} \\]

**Why are these metrics important?**

In email spam detection, accuracy alone is not sufficient.

For example, if a genuine email is incorrectly classified as Spam, the user might miss an important message.

Therefore, Precision and Recall are also important.

**9. Performance Comparison of KNN and SVM**

The following is an illustrative comparison. Actual results depend on the dataset, feature extraction, and model parameters.

| Evaluation Metric | KNN   | SVM   |
| ----------------- | ----- | ----- |
| Accuracy          | 90%   | 96%   |
| Precision         | 88%   | 95%   |
| Recall            | 89%   | 94%   |
| F1-Score          | 88.5% | 94.5% |

**Analysis:**

- KNN classifies emails based on the similarity of nearby training examples.
- SVM identifies a decision boundary that separates the two classes.
- SVM may perform better on high-dimensional email text features.
- KNN may require more prediction time for larger datasets.
- SVM generally provides better classification performance in this illustrative comparison.

Therefore, SVM is the better-performing algorithm in the above example. However, the actual winner must be determined using experimental results.

**10. Steps to Perform the Assignment**

Understand this workflow because it represents the actual implementation of your practical.

1. **Import Libraries**

   Import Pandas, NumPy and Scikit-learn.
2. **Load Dataset**

   Load the email dataset containing Spam and Not Spam labels.
3. **Preprocess Data**

   Remove unnecessary data, clean email text and handle missing values.
4. **Feature Extraction**

   Convert email text into numerical features using TF-IDF.
5. **Split Dataset**

   Divide the dataset into training and testing sets.
6. **Train KNN Model**

   Train the KNN classifier using the training data.
7. **Train SVM Model**

   Train the SVM classifier using the same training data.
8. **Make Predictions**

   Predict Spam or Not Spam for test emails.
9. **Evaluate Performance**

   Calculate accuracy, precision, recall, F1-score and confusion matrix.
10. **Compare Results**

    Identify which algorithm performs better.

**11. Important Python Libraries**

| Library               | Purpose                                    |
| --------------------- | ------------------------------------------ |
| Pandas                | Loading and manipulating datasets          |
| NumPy                 | Numerical operations                       |
| Scikit-learn          | Machine learning algorithms                |
| TfidfVectorizer       | Converting text into numerical features    |
| train_test_split      | Splitting training and testing data        |
| KNeighborsClassifier  | Implementing KNN                           |
| SVC                   | Implementing SVM                           |
| accuracy_score        | Calculating accuracy                       |
| confusion_matrix      | Generating confusion matrix                |
| classification_report | Calculating precision, recall and F1-score |

**12. Important Viva Questions and Answers**

**Q1. What is spam detection?**

Spam detection is a machine learning application used to identify unwanted emails and separate them from genuine emails.

**Q2. Why is spam detection a binary classification problem?**

Because it classifies emails into two possible categories: Spam and Not Spam.

**Q3. Which type of machine learning is used?**

Supervised machine learning because the model is trained using labelled email data.

**Q4. What is KNN?**

K-Nearest Neighbors is an algorithm that classifies a new data point based on the majority class of its K nearest neighbors.

**Q5. What does K represent in KNN?**

K represents the number of nearest neighbors considered during classification.

**Q6. What is SVM?**

Support Vector Machine is a supervised learning algorithm that separates different classes using an optimal hyperplane.

**Q7. What are support vectors?**

Support vectors are the data points closest to the decision boundary. They help determine the position of the hyperplane.

**Q8. What is a hyperplane?**

A hyperplane is a decision boundary that separates data points belonging to different classes.

**Q9. What is TF-IDF?**

TF-IDF stands for Term Frequency–Inverse Document Frequency. It converts text into numerical values based on the importance of words in documents.

**Q10. Which algorithm is better for spam detection?**

SVM often performs well for high-dimensional text classification, but the actual performance depends on the dataset and model parameters.

**Q11. What is overfitting?**

Overfitting occurs when a machine learning model learns the training data too closely, including noise, and performs poorly on unseen data.

**Q12. Why do we split data into training and testing sets?**

Training data is used to teach the model, whereas testing data is used to evaluate its performance on unseen examples.

**13. Conclusion**

Email Spam Detection is a binary classification problem that classifies incoming emails into Normal (Not Spam) and Abnormal (Spam) categories.

In this assignment, two supervised machine learning algorithms, K-Nearest Neighbors (KNN) and Support Vector Machine (SVM), are used for classification.

KNN classifies emails based on their nearest neighbors, whereas SVM classifies emails by finding an optimal decision boundary.

Their performance is evaluated using accuracy, precision, recall, F1-score and confusion matrix. The comparison helps determine the more suitable algorithm for email spam detection.

**My recommendation for learning this assignment**

Understand these five concepts particularly well before attempting the practical:

1. Binary Classification and Supervised Learning.
2. Feature Extraction and TF-IDF.
3. KNN – How distance and majority voting work.
4. SVM – Hyperplane, margin and support vectors.
5. Confusion Matrix – TP, TN, FP, FN and evaluation metrics.

Once these five concepts are clear, the entire assignment becomes much easier to understand rather than simply memorizing the theory.