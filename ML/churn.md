# **Bank Customer Churn Prediction Using Artificial Neural Network (ANN)**

Let's understand this assignment from the basics, just like the previous spam detection assignment.

The main objective is to build a **Neural Network-based classification model** that predicts whether a bank customer will leave the bank within the next six months.

This assignment introduces an important concept in Machine Learning called **Artificial Neural Networks (ANN)**.

We will understand:

1. What is customer churn?
2. What is binary classification?
3. What is an Artificial Neural Network?
4. How does a neural network work?
5. Dataset and feature selection.
6. Training and testing data.
7. Data normalization.
8. Building and improving the neural network.
9. Accuracy score and confusion matrix.
10. Important viva questions.

**1. Introduction to Customer Churn Prediction**

**What is Customer Churn?**

Customer Churn refers to a situation in which a customer stops using the services of a company or organization.

In banking, customer churn occurs when a customer closes their bank account or stops using the bank's services and moves to another bank.

For example:

Suppose a bank has 10,000 customers. Out of these:

- 8,000 customers continue using the bank.
- 2,000 customers leave the bank.

The bank wants to identify which existing customers are likely to leave within the next six months.

This is called **Customer Churn Prediction**.

**Why is Customer Churn Prediction important?**

1. Customer Retention: Helps banks retain existing customers.
2. Business Growth: Reduces customer loss and improves profitability.
3. Better Services: Helps identify customers who need better banking services.
4. Targeted Offers: Banks can provide special offers to customers who may leave.
5. Decision Making: Helps management make data-driven decisions.

**2. What is Binary Classification?**

Binary Classification is a supervised machine learning technique used to classify data into two possible categories.

In this assignment, the target variable is called `Exited`.

It has two possible values:

| Exited Value | Meaning                      | Classification |
| ------------ | ---------------------------- | -------------- |
| 0            | Customer stays with the bank | Not Churn      |
| 1            | Customer leaves the bank     | Churn          |

**Example:**

| Customer ID | Age | Credit Score | Exited |
| ----------- | --- | ------------ | ------ |
| 15634602    | 42  | 619          | 1      |
| 15647311    | 41  | 608          | 0      |
| 15619304    | 42  | 502          | 1      |
| 15701354    | 39  | 699          | 0      |

The model learns from these existing customer records and predicts whether a new customer will leave or stay.

**Input: Bank Customer Data**

Age, Credit Score, Balance, Geography, Gender, etc.

**Artificial Neural Network**

Processes customer information and learns patterns

**0 – Not Churn**

Customer stays

**1 – Churn**

Customer leaves

**3. What is an Artificial Neural Network (ANN)?**

An Artificial Neural Network is a machine learning model inspired by the working of the human brain.

The human brain contains neurons that communicate with each other. Similarly, an ANN contains artificial neurons that process information and learn patterns from data.

ANN is used for:

- Classification problems.
- Prediction problems.
- Pattern recognition.
- Image and speech recognition.
- Customer churn prediction.

**Basic Structure of ANN**

An Artificial Neural Network consists of three main layers:

1. Input Layer
2. Hidden Layer
3. Output Layer

![alt text](image.png)

[yjjo.tistory.com](https://yjjo.tistory.com/5?utm_source=chatgpt.com)

**A. Input Layer**

- It is the first layer of the neural network.
- It receives input data from the dataset.
- Each neuron represents an input feature.

For example, customer age, credit score, balance, tenure, etc.

**B. Hidden Layer**

- It is the middle layer of the neural network.
- It processes the input information.
- It contains neurons that learn patterns from the data.
- It uses weights, biases and activation functions.

A neural network can have one or multiple hidden layers.

**C. Output Layer**

- It is the final layer of the neural network.
- It produces the prediction.
- In our assignment, it contains one neuron because we have two possible output classes.

Output:

- 0 = Customer will stay.
- 1 = Customer will leave.

**How does an Artificial Neural Network work?**

Let's understand with an example.

Suppose a customer has the following information:

- Age = 45
- Credit Score = 550
- Balance = ₹1,50,000
- Tenure = 2 years

The ANN receives this information through the input layer.

The hidden layer processes these values and identifies patterns based on its training.

Finally, the output layer predicts the probability of customer churn.

For example:

**Illustrative prediction**

# **82%**

Predicted probability of customer churn

Predicted class: 1 – Churn

The model predicts churn because the probability is greater than the classification threshold of 0.5.

**4. Important Concepts Used in ANN**

These concepts are important for understanding how a neural network learns.

**A. Neurons**

A neuron is the basic processing unit of an artificial neural network.

It receives inputs, performs mathematical calculations and produces an output.

**B. Weights**

Weights determine the importance of each input feature.

For example, age might have a different influence on churn prediction than customer tenure.

The network adjusts weights during training to improve predictions.

**C. Bias**

Bias is an additional value added to the weighted sum of inputs.

It helps the neuron adjust its output independently of the input values.

**D. Activation Function**

An activation function determines whether and how strongly a neuron should activate.

It introduces non-linearity into the neural network, allowing it to learn complex patterns.

Important activation functions:

| Function | Purpose                                            |
| -------- | -------------------------------------------------- |
| ReLU     | Commonly used in hidden layers                     |
| Sigmoid  | Converts output into a probability between 0 and 1 |
| Tanh     | Produces values between -1 and +1                  |
| Softmax  | Produces class probabilities for multiple classes  |

For our assignment:

- Hidden layer: ReLU
- Output layer: Sigmoid

**Why use ReLU?**

ReLU stands for Rectified Linear Unit.

\\[ f(x)=\max(0,x) \\]

It returns zero for negative values and the input itself for positive values.

**Why use Sigmoid?**

Sigmoid converts the output into a value between 0 and 1.

\\[ f(x)=\frac{1}{1+e^{-x}} \\]

For example:

- 0.85 means 85% predicted probability of churn.
- 0.20 means 20% predicted probability of churn.

If the output is greater than or equal to 0.5, we classify it as Churn (1). Otherwise, it is Not Churn (0).

**E. Forward Propagation**

Forward propagation is the process of passing input data through the neural network to generate a prediction.

Steps:

1. Input features enter the input layer.
2. Hidden layers perform calculations.
3. Activation functions are applied.
4. The output layer generates a prediction.

**F. Backpropagation**

Backpropagation is the process of calculating errors and updating the network's weights and biases to improve its predictions.

Steps:

1. Generate a prediction.
2. Compare the prediction with the actual output.
3. Calculate the prediction error.
4. Calculate gradients using backpropagation.
5. Update weights and biases.
6. Repeat the process until the model improves.

**G. Epoch**

An epoch represents one complete pass of the entire training dataset through the neural network.

For example, if we set epochs = 50, the model processes the training dataset 50 times.

**H. Batch Size**

Batch size represents the number of training samples processed before updating the model's weights.

For example, batch size = 32 means the model processes 32 samples before performing a weight update.

**I. Loss Function**

A loss function measures the difference between the actual output and the predicted output.

For binary classification, we commonly use Binary Cross-Entropy.

\\[ L=-[y\log(p)+(1-y)\log(1-p)] \\]

Where:

- \\(y\\) = Actual output.
- \\(p\\) = Predicted probability.

The objective is to minimize the loss.

**J. Optimizer**

An optimizer updates the weights and biases to minimize the loss function.

Adam (Adaptive Moment Estimation) is a commonly used optimizer for neural networks.

**5. Understanding the Dataset**

Our assignment uses the Bank Customer Churn Modeling dataset from Kaggle.

Source: [Bank Customer Churn Modeling – Kaggle](https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling?utm_source=chatgpt.com).

The dataset contains approximately 10,000 customer records and 14 columns.

Let's understand the features.

| Feature         | Description                               |
| --------------- | ----------------------------------------- |
| RowNumber       | Serial number of the record               |
| CustomerId      | Unique customer identification            |
| Surname         | Customer's surname                        |
| CreditScore     | Customer's credit score                   |
| Geography       | Customer's country                        |
| Gender          | Male or Female                            |
| Age             | Customer's age                            |
| Tenure          | Number of years with the bank             |
| Balance         | Bank account balance                      |
| NumOfProducts   | Number of bank products used              |
| HasCrCard       | Whether the customer has a credit card    |
| IsActiveMember  | Whether the customer is an active member  |
| EstimatedSalary | Estimated annual salary                   |
| Exited          | Target variable indicating customer churn |

**What is the target variable?**

`Exited` is the target variable.

- 0 = Customer did not leave the bank.
- 1 = Customer left the bank.

**What are features?**

Features are the input variables used by the machine learning model to make predictions.

For example:

CreditScore, Geography, Gender, Age, Tenure, Balance, etc.

**Important:** RowNumber, CustomerId and Surname are generally removed before training because they are identifiers or names, not useful general predictive features.

**6. Step-by-Step Implementation of the Assignment**

Now let's understand every practical step.

**Step 1: Read the Dataset**

First, we load the dataset using the Pandas library.

```
import pandas as pddataset = pd.read_csv("Churn_Modelling.csv")print(dataset.head())print(dataset.shape)print(dataset.info())
```

Explanation:

- `import pandas`: Imports the Pandas library.
- `read_csv()`: Reads the CSV dataset.
- `head()`: Displays the first five records.
- `shape`: Displays the number of rows and columns.
- `info()`: Displays column names and data types.

Expected shape:

```
(10000, 14)
```

**Step 2: Distinguish Features and Target Variables**

We separate the independent variables (X) and dependent variable (Y).

**Independent Variables (X):**

These are the input features used to predict customer churn.

**Dependent Variable (Y):**

This is the output that we want to predict.

```
X = dataset.drop(    ['RowNumber', 'CustomerId', 'Surname', 'Exited'],    axis=1)y = dataset['Exited']
```

Explanation:

- `X` contains the input features.
- `y` contains the target variable.
- `drop()` removes unnecessary columns.

After removing the three identifiers, we have 10 input columns and one target column.

**Step 3: Convert Categorical Data into Numerical Data**

Neural networks work with numerical inputs.

However, our dataset contains categorical columns such as Geography and Gender.

We convert these columns into numerical form using One-Hot Encoding.

```
X = pd.get_dummies(    X,    columns=['Geography', 'Gender'],    drop_first=True,    dtype=int)
```

Example:

Before encoding:

| Geography |
| --------- |
| France    |
| Germany   |
| Spain     |

After encoding:

| Geography_Germany | Geography_Spain |
| ----------------- | --------------- |
| 0                 | 0               |
| 1                 | 0               |
| 0                 | 1               |

France becomes the reference category.

**Step 4: Divide the Dataset into Training and Testing Sets**

We divide the dataset into two parts:

1. Training Set: Used to train the neural network.
2. Testing Set: Used to evaluate the trained model.

Generally, we use an 80:20 ratio.

- 80% training data = 8,000 records.
- 20% testing data = 2,000 records.

```
from sklearn.model_selection import train_test_splitX_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.2,    random_state=42,    stratify=y)
```

Explanation:

- `test_size=0.2`: Allocates 20% data for testing.
- `random_state=42`: Ensures reproducible splitting.
- `stratify=y`: Maintains approximately the same class proportions in both sets.

**Step 5: Normalize the Training and Testing Data**

Normalization is the process of bringing numerical features into a similar scale.

Why is normalization required?

Consider:

- Age = 40
- Credit Score = 700
- Balance = 1,50,000

These features have very different numerical ranges.

Large numerical values may dominate the learning process, so scaling helps the neural network learn more effectively.

We use StandardScaler.

Formula:

\\[ Z=\frac{X-\mu}{\sigma} \\]

Where:

- \\(X\\) = Original value.
- \\(\mu\\) = Mean of the feature.
- \\(\sigma\\) = Standard deviation.
- \\(Z\\) = Standardized value.

Python code:

```
from sklearn.preprocessing import StandardScalersc = StandardScaler()X_train = sc.fit_transform(X_train)X_test = sc.transform(X_test)
```

**Important difference:**

- `fit_transform()` calculates the mean and standard deviation from training data and transforms it.
- `transform()` applies the same training-data scaling to testing data.

We should never fit the scaler separately on testing data because that causes data leakage.

**Step 6: Initialize and Build the Neural Network Model**

Now we create our Artificial Neural Network using TensorFlow and Keras.

First, understand the architecture.

For this assignment, we can use:

| Layer          | Number of Neurons | Activation |
| -------------- | ----------------- | ---------- |
| Input Layer    | 12                | —          |
| Hidden Layer 1 | 16                | ReLU       |
| Hidden Layer 2 | 8                 | ReLU       |
| Output Layer   | 1                 | Sigmoid    |

Why 12 input neurons?

After removing three unnecessary columns and encoding Geography and Gender, we get 12 input features.

**Python Code:**

```
import tensorflow as tffrom tensorflow.keras.models import Sequentialfrom tensorflow.keras.layers import Densemodel = Sequential()# First hidden layermodel.add(Dense(16, activation='relu',                input_shape=(X_train.shape[1],)))# Second hidden layermodel.add(Dense(8, activation='relu'))# Output layermodel.add(Dense(1, activation='sigmoid'))
```

Explanation:

- `Sequential()`: Creates a neural network layer by layer.
- `Dense(16)`: Creates a fully connected layer with 16 neurons.
- `activation='relu'`: Applies ReLU activation.
- `input_shape`: Specifies the number of input features.
- `Dense(1)`: Creates one output neuron.
- `activation='sigmoid'`: Produces a probability between 0 and 1.

**Step 7: Compile the Model**

After building the model, we compile it.

Compilation specifies the loss function, optimizer and evaluation metrics.

```
model.compile(    optimizer='adam',    loss='binary_crossentropy',    metrics=['accuracy'])
```

Explanation:

| Parameter            | Purpose                      |
| -------------------- | ---------------------------- |
| Adam                 | Optimizes model weights      |
| Binary Cross-Entropy | Measures prediction error    |
| Accuracy             | Measures correct predictions |

**Step 8: Train the Model**

We train our neural network using the training dataset.

```
history = model.fit(    X_train,    y_train,    epochs=50,    batch_size=32,    validation_split=0.2,    verbose=1)
```

Explanation:

- `X_train`: Training input features.
- `y_train`: Actual training labels.
- `epochs=50`: Trains the model for 50 epochs.
- `batch_size=32`: Processes 32 records per batch.
- `validation_split=0.2`: Uses 20% of the training data for validation.
- `verbose=1`: Displays training progress.

**What happens during training?**

1. The model receives customer data.
2. It predicts whether the customer will leave.
3. The prediction is compared with the actual result.
4. The loss function calculates the error.
5. Backpropagation calculates gradients.
6. Adam updates the weights.
7. The process repeats for multiple epochs.

**Step 9: Identify Points of Improvement**

After building the initial model, we can improve its performance.

Some important improvement techniques are:

| Technique               | Purpose                                                    |
| ----------------------- | ---------------------------------------------------------- |
| Increase hidden neurons | Helps the model learn more complex patterns                |
| Add hidden layers       | Learns more complex relationships                          |
| Dropout                 | Reduces overfitting                                        |
| Early Stopping          | Stops training when validation performance stops improving |
| Adjust learning rate    | Controls the speed of weight updates                       |
| Class Weighting         | Helps handle imbalanced classes                            |

**What is Overfitting?**

Overfitting occurs when a model performs very well on training data but performs poorly on unseen testing data.

For example:

- Training accuracy = 99%
- Testing accuracy = 78%

This indicates that the model may have memorized training data instead of learning general patterns.

**How can we reduce overfitting?**

We can use Dropout and Early Stopping.

**A. Dropout**

Dropout is a regularization technique that randomly disables some neurons during training.

For example, dropout = 0.3 means approximately 30% of the selected layer's neurons are temporarily disabled during each training update.

It helps prevent the model from depending too heavily on particular neurons.

**B. Early Stopping**

Early Stopping monitors the validation loss during training.

If validation loss does not improve for a specified number of epochs, training stops automatically.

**Improved Neural Network Code**

```
from tensorflow.keras.layers import Dropoutfrom tensorflow.keras.callbacks import EarlyStoppingmodel = Sequential([    Dense(32, activation='relu',          input_shape=(X_train.shape[1],)),    Dropout(0.3),    Dense(16, activation='relu'),    Dropout(0.2),    Dense(1, activation='sigmoid')])model.compile(    optimizer='adam',    loss='binary_crossentropy',    metrics=['accuracy'])early_stop = EarlyStopping(    monitor='val_loss',    patience=5,    restore_best_weights=True)history = model.fit(    X_train,    y_train,    epochs=100,    batch_size=32,    validation_split=0.2,    callbacks=[early_stop],    verbose=1
```

This improved model has:

- More neurons in the first hidden layer.
- Two hidden layers.
- Dropout regularization.
- Early stopping to prevent unnecessary training.

Note that a more complex model does not automatically mean better accuracy. We should compare its performance against the initial model on the same test set.

**7. Accuracy Score and Confusion Matrix**

After training the neural network, we evaluate its performance using testing data.

**A. Prediction**

```
y_pred = model.predict(X_test)y_pred = (y_pred >= 0.5).astype(int)
```

Explanation:

The model produces probabilities between 0 and 1.

We convert them into binary classes:

- Probability greater than or equal to 0.5 = 1 (Churn).
- Probability less than 0.5 = 0 (Not Churn).

**B. Accuracy Score**

Accuracy measures how many predictions are correct out of all predictions.

\\[ Accuracy=\frac{Correct\ Predictions}{Total\ Predictions}\times100 \\]

Python code:

```
from sklearn.metrics import accuracy_scoreaccuracy = accuracy_score(y_test, y_pred)print("Accuracy:", accuracy)print("Accuracy Percentage:", accuracy * 100)
```

**C. Confusion Matrix**

A confusion matrix is used to evaluate the classification performance of a model.

It compares actual results with predicted results.

```
from sklearn.metrics import confusion_matrixcm = confusion_matrix(y_test, y_pred)print("Confusion Matrix:")print(cm)
```

Example confusion matrix (illustrative):

**Confusion Matrix**

Rows = Actual class | Columns = Predicted class

Predicted 0

Predicted 1

Actual 0

# **1,540**

True Negative

# **60**

False Positive

Actual 1

# **100**

False Negative

# **300**

True Positive

Illustrative values only, not actual model output.

Understand the four terms:

1. True Positive (TP): Customer actually left and the model correctly predicted churn.
2. True Negative (TN): Customer stayed and the model correctly predicted no churn.
3. False Positive (FP): Customer stayed but the model predicted churn.
4. False Negative (FN): Customer left but the model predicted no churn.

**Why is the confusion matrix important?**

Suppose a bank wants to identify customers who are likely to leave.

If the model incorrectly predicts that a customer will stay when they actually leave, the bank loses an opportunity to retain that customer.

Therefore, we should also evaluate Recall and Precision, not just accuracy.

**D. Other Important Evaluation Metrics**

Apart from accuracy, we can calculate Precision, Recall and F1-score.

```
from sklearn.metrics import classification_reportprint(classification_report(y_test, y_pred))
```

| Metric    | Meaning                                                   |
| --------- | --------------------------------------------------------- |
| Precision | How many predicted churn customers actually left          |
| Recall    | How many actual churn customers were correctly identified |
| F1-score  | Balance between precision and recall                      |

**Important observation:** The dataset is imbalanced because approximately 20% of customers belong to the churn class and 80% belong to the non-churn class. Therefore, a model predicting every customer as Not Churn could still achieve approximately 80% accuracy.

This is why Recall, F1-score and the confusion matrix are important.

**8. Complete Flow of the Assignment**

Remember this sequence for your practical examination:

1. Read the bank customer dataset using Pandas.
2. Separate independent features (X) and dependent target (Y).
3. Remove unnecessary columns and encode categorical data.
4. Split the dataset into training and testing sets.
5. Normalize data using StandardScaler.
6. Initialize the Artificial Neural Network.
7. Build input, hidden and output layers.
8. Compile the model using Adam and Binary Cross-Entropy.
9. Train the model using training data.
10. Improve the model using Dropout and Early Stopping.
11. Predict customer churn using testing data.
12. Calculate accuracy and confusion matrix.

**9. Important Viva Questions and Answers**

These are the questions you should prepare for your practical examination.

**Q1. What is customer churn?**

Customer churn refers to a situation where a customer stops using the services of a company or bank.

**Q2. What is ANN?**

Artificial Neural Network is a machine learning model inspired by the human brain. It consists of interconnected artificial neurons that learn patterns from data.

**Q3. What are the three main layers of ANN?**

- Input Layer
- Hidden Layer
- Output Layer

**Q4. Why is ANN used in this assignment?**

ANN is used to learn patterns from customer information and predict whether a customer will leave the bank or continue using its services.

**Q5. What is the target variable?**

The target variable is `Exited`, which represents whether a customer has left the bank.

**Q6. What is the difference between features and target variables?**

Features are the input variables used for prediction, whereas the target variable is the output that the model predicts.

**Q7. Why do we normalize data?**

Normalization brings features into a similar numerical scale and helps neural networks train more effectively.

**Q8. What is StandardScaler?**

StandardScaler is a preprocessing technique that transforms data to have approximately zero mean and unit standard deviation.

**Q9. What is an activation function?**

An activation function determines the output of a neuron and introduces non-linearity into a neural network.

**Q10. Why do we use ReLU?**

ReLU is commonly used in hidden layers because it is computationally simple and helps neural networks learn non-linear patterns.

**Q11. Why do we use Sigmoid?**

Sigmoid converts the output into a probability between 0 and 1, making it suitable for binary classification.

**Q12. What is backpropagation?**

Backpropagation calculates gradients of the loss with respect to network parameters and helps update weights and biases to reduce prediction errors.

**Q13. What is an epoch?**

One epoch represents one complete pass of the training dataset through the neural network.

**Q14. What is Dropout?**

Dropout is a regularization technique that randomly disables selected neurons during training to reduce overfitting.

**Q15. What is Early Stopping?**

Early Stopping stops training when the validation performance stops improving for a specified number of epochs.

**Q16. Why do we use Adam optimizer?**

Adam is an optimization algorithm that adjusts neural network weights efficiently using adaptive learning rates.

**Q17. What is Binary Cross-Entropy?**

It is a loss function used to measure the difference between actual and predicted probabilities in binary classification.

**Q18. What is a confusion matrix?**

A confusion matrix is a table that compares actual and predicted classes using True Positive, True Negative, False Positive and False Negative.

**Q19. What is overfitting?**

Overfitting occurs when a model performs well on training data but poorly on unseen data.

**Q20. What is the difference between training and testing data?**

Training data is used to teach the model, while testing data is used to evaluate its performance.

**10. Conclusion (Journal Answer)**

The Bank Customer Churn Prediction system is developed using an Artificial Neural Network (ANN). The dataset contains 10,000 customer records with different features such as Credit Score, Geography, Gender, Age, Tenure, Balance and Estimated Salary.

The dataset is preprocessed by separating input and target variables, encoding categorical data, splitting the dataset into training and testing sets, and normalizing the features.

An ANN model is constructed using input, hidden and output layers. The model is trained using the Adam optimizer and Binary Cross-Entropy loss function. Techniques such as Dropout and Early Stopping can be implemented to improve model performance and reduce overfitting.

Finally, the model predicts whether a customer will leave the bank or stay. Its performance is evaluated using the accuracy score and confusion matrix.

Thus, the system helps banks identify customers who may leave and take appropriate customer retention measures.

**11. The Most Important Concepts to Understand**

Here is how I recommend studying this assignment. Don't memorize all 20 viva answers immediately.

| Priority | Concept                | What you should understand               |
| -------- | ---------------------- | ---------------------------------------- |
| 1        | ANN                    | How input, hidden and output layers work |
| 2        | Neurons, Weights, Bias | How a neural network learns              |
| 3        | ReLU and Sigmoid       | Why activation functions are required    |
| 4        | Forward Propagation    | How predictions are generated            |
| 5        | Backpropagation        | How errors are reduced                   |
| 6        | Normalization          | Why features need scaling                |
| 7        | Epoch and Batch Size   | How model training works                 |
| 8        | Dropout                | How overfitting is reduced               |
| 9        | Adam Optimizer         | How weights are updated                  |
| 10       | Confusion Matrix       | How model performance is evaluated       |

**One important distinction for your understanding:**

The assignment describes predicting whether a customer will leave in the next six months. However, the Kaggle dataset's `Exited` column represents whether the customer exited in the recorded data; it does not itself provide a clearly defined six-month prediction horizon. A genuine six-month prediction system would require appropriately time-labelled historical customer data.

This distinction is useful if your examiner asks about the dataset.

**My suggestion:** First understand how a single artificial neuron works (inputs → weights → bias → activation → output). Once you understand that, hidden layers, forward propagation, backpropagation and the complete ANN become much easier.