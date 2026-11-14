# **K-Means Clustering / Hierarchical Clustering Using Sales Dataset**

Let's understand this assignment from the absolute basics, including the theory, working, mathematical concepts, Python implementation and viva questions.

The main objective is to **group similar sales records into different clusters using unsupervised machine learning** and determine the optimal number of clusters using the Elbow Method.

We will focus on K-Means first and then understand Hierarchical Clustering.

**1. Introduction to Clustering**

**What is Clustering?**

Clustering is an unsupervised machine learning technique used to group similar data points into different groups called clusters.

Data points within the same cluster have similar characteristics, whereas data points belonging to different clusters have different characteristics.

In simple words, clustering means grouping similar things together.

**Example:**

Suppose a shopping company has customer sales information.

It wants to divide customers into groups based on their purchasing behaviour.

- Cluster 1: Customers with low sales.
- Cluster 2: Customers with medium sales.
- Cluster 3: Customers with high sales.

This process is called Clustering.

[K-Means Clustering : What I Learned In Unsupervised Learning | by Aniket | Feb, 2026 | Medium](https://images.openai.com/static-rsc-4/3v1W6Q5JchPbzaluk6RuEm1ht_MJhFf2QohPRQCwbhuGGhMY28dTZkN9fwkAgjhqBs2DiVLz0yF9OAjbhRePcy982cQdxZwKNHEOTzRmdQvwh_1yjxiMz4oOtDpIxW1loQ8tKjVHp7Hn57rSy1yauK7QBLDOtOMhnBkHbob-Dpg?purpose=inline)

[medium.com](https://medium.com/%40aniani2521/k-means-clustering-what-i-learned-in-unsupervised-learning-e83bfc3ce4bf?utm_source=chatgpt.com)

**Why is Clustering used?**

1. Customer Segmentation: Group customers based on purchasing behaviour.
2. Sales Analysis: Identify different sales patterns.
3. Marketing: Create targeted marketing campaigns.
4. Business Decisions: Understand customer preferences.
5. Pattern Recognition: Discover hidden patterns in datasets.

**2. What is Unsupervised Machine Learning?**

Unsupervised Learning is a type of machine learning in which the model learns patterns from data without predefined output labels.

Unlike supervised learning, the dataset does not contain predefined answers.

For example:

In supervised learning, we provide customer data along with labels such as High Sales or Low Sales.

In unsupervised learning, we provide only customer data, and the algorithm discovers groups automatically.

**Types of Unsupervised Learning**

1. Clustering: Groups similar data points together.
2. Association: Identifies relationships between different items.
3. Dimensionality Reduction: Reduces the number of features while preserving important information.

Our assignment uses Clustering.

**3. Understanding the Sales Dataset**

The assignment uses the `sales_data_sample.csv` dataset.

This dataset contains sales transaction information, including details such as:

| Feature         | Description                  |
| --------------- | ---------------------------- |
| ORDERNUMBER     | Unique order identification  |
| QUANTITYORDERED | Quantity of products ordered |
| PRICEEACH       | Price of each product        |
| SALES           | Total sales amount           |
| ORDERDATE       | Date of the order            |
| STATUS          | Order status                 |
| PRODUCTLINE     | Category of the product      |
| CUSTOMERNAME    | Name of the customer         |
| COUNTRY         | Customer's country           |
| DEALSIZE        | Size of the deal             |

**What are features in clustering?**

Features are the attributes used by the clustering algorithm to identify similarities between data points.

For example, we can use:

- Quantity Ordered
- Price Each
- Sales

The algorithm uses these numerical features to group similar sales records.

Note that columns such as ORDERNUMBER and CUSTOMERNAME are identifiers or names and should not normally be treated as numerical clustering features.

**4. K-Means Clustering Algorithm**

Main algorithm

**What is K-Means Clustering?**

K-Means is an unsupervised machine learning algorithm used to divide a dataset into K different clusters based on similarities between data points.

Here, K represents the number of clusters we want to create.

The algorithm groups data points by calculating their distance from the cluster centers, called **Centroids**.

In simple words:

K-Means finds K centers and groups every data point with its nearest center.

**Important Terms in K-Means**

**A. K (Number of Clusters)**

K represents the number of groups into which we want to divide the dataset.

For example:

- K = 2 means two clusters.
- K = 3 means three clusters.
- K = 4 means four clusters.

**B. Centroid**

A centroid is the central point of a cluster.

It represents the average position of all data points belonging to that cluster.

**C. Distance**

Distance measures how far one data point is from another.

K-Means commonly uses Euclidean Distance.

\\[ d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2} \\]

**D. Inertia (WCSS)**

WCSS stands for Within-Cluster Sum of Squares.

It measures the sum of squared distances between each data point and its assigned cluster centroid.

\\[ WCSS=\sum\_{i=1}^{K}\sum\_{x\in C_i} ||x-\mu_i||^2 \\]

Where:

- K = Number of clusters.
- \\(C_i\\) = Data points belonging to cluster i.
- \\(x\\) = Individual data point.
- \\(\mu_i\\) = Centroid of the cluster.

Lower WCSS means data points are closer to their respective centroids.

**How does K-Means work?**

K-Means follows these steps:

1. Select the number of clusters (K).
2. Randomly initialize K centroids.
3. Calculate the distance between each data point and all centroids.
4. Assign each data point to its nearest centroid.
5. Recalculate the centroid by taking the mean of all points in that cluster.
6. Repeat the assignment and centroid calculation.
7. Stop when the centroids no longer change significantly or the maximum number of iterations is reached.

**Example of K-Means**

Suppose we have 9 customers with different sales values.

Initially, K-Means selects three centroids.

It assigns customers to the nearest centroid and creates three groups.

[k-means Clustering: Math Explained & Multi-Dimension Implementations | by Salah Assana | Medium](https://images.openai.com/static-rsc-4/h4LvIMKC61jb5JpetOPQMtKAbdlbX0S9TUOO8Ts4-Jd2mAGGW_qHDcEMoEL7N8pBcKcwuD681elSKzj-aaFQr_kBgCb3hBD-FeoaFNt5Gau3N_LmMBNQKyQNycvBxEdy2N0j2CaDOCpm35OqdupITIv3CNrUKoYBynzEYW8rkpA?purpose=inline)

[medium.com](https://medium.com/%40SalahAssana/k-means-clustering-math-explained-multi-dimension-implementations-85e202bc2ec0?utm_source=chatgpt.com)

After several iterations, the clusters become stable.

The final output contains three different groups of customers based on their sales behaviour.

**Advantages of K-Means**

1. Simple and easy to understand.
2. Easy to implement.
3. Works efficiently on large datasets.
4. Produces clear and distinct clusters when the data is suitable.
5. Useful for customer segmentation and sales analysis.

**Disadvantages of K-Means**

1. The number of clusters K must be specified.
2. Sensitive to initial centroid selection.
3. Sensitive to outliers.
4. Requires numerical data.
5. May not perform well when clusters have irregular shapes or very different densities.

**5. Elbow Method**

Important part of the assignment

**What is the Elbow Method?**

The Elbow Method is a technique used to determine the optimal number of clusters (K) in K-Means clustering.

The main purpose is to identify a suitable K value that minimizes WCSS without creating unnecessary clusters.

**Why do we need the Elbow Method?**

Suppose we have a sales dataset.

We do not know whether the dataset should be divided into:

- 2 clusters
- 3 clusters
- 4 clusters
- 5 clusters

If we select too few clusters, different types of sales records may be grouped together.

If we select too many clusters, the data may be divided into unnecessary small groups.

The Elbow Method helps us find a suitable number of clusters.

**How does the Elbow Method work?**

1. Select different values of K, such as 1 to 10.
2. Apply K-Means for each value of K.
3. Calculate WCSS (inertia) for every model.
4. Plot K values on the X-axis.
5. Plot WCSS values on the Y-axis.
6. Identify the point where the decrease in WCSS starts slowing down significantly.

This point looks like an elbow in the graph.

[Weather Typing and Synoptic Climatology – CEVE 543 - Fall 2025](https://images.openai.com/static-rsc-4/97VOIqvuPECxNORhms5bM4QV9NeaktnYZiycPNHBH6PL3_5v4Rle-GnvA58isbQCcofvhYG6LRnKdRAdRsDZueoJSbxbYSHVQOwI3TN2ofRMf40prS4MaJ65Cj-wrLvIKWJziThEcvH8Qpo9hdmHwZTsf-5P4q-3GlYHGYEN-tI?purpose=inline)

[ceve543.github.io](https://ceve543.github.io/lectures/2025-10-27/index.html?utm_source=chatgpt.com)

**Understanding the Elbow Graph**

- X-axis: Number of clusters (K).
- Y-axis: WCSS (Within-Cluster Sum of Squares).

As K increases, WCSS decreases because data points become closer to their assigned centroids.

However, after a particular point, increasing K produces only a small improvement.

That point is called the Elbow Point.

For example, if the elbow appears at K = 3, we can select three clusters.

**Important:** The Elbow Method gives an estimate, not always a perfectly clear answer. Sometimes the graph does not have a sharp elbow.

**6. Hierarchical Clustering**

Alternative algorithm

**What is Hierarchical Clustering?**

Hierarchical Clustering is an unsupervised machine learning algorithm that groups similar data points into a hierarchy of clusters.

Unlike K-Means, it does not require us to specify the number of clusters at the beginning.

It creates a tree-like structure called a **Dendrogram**.

**Types of Hierarchical Clustering**

There are two main types:

**A. Agglomerative Clustering (Bottom-Up Approach)**

Agglomerative Clustering is the most commonly used hierarchical clustering technique.

It works by initially treating every data point as an individual cluster.

Then, it repeatedly merges the closest clusters until all points are combined into one cluster or the desired number of clusters is reached.

Steps:

1. Initially, each data point forms a separate cluster.
2. Calculate distances between clusters.
3. Merge the two closest clusters.
4. Recalculate distances.
5. Repeat the process until the stopping condition is reached.

[An Introduction to Hierarchical Clustering in Python | DataCamp](https://images.openai.com/static-rsc-4/xJtfIudr_LdI_mmEgvZxVKHFoAFp99Lo1E6e9g4wtJdgqESMuc3nqS9O8KWTowi_FO5FS1tK8PBc_QEWrSEPej4MiRRHyGDZwVW9V9cluCgG60VNNsPG5lxrvMoJUTAXPkkQATTDXMxSRKo2WJf2TfT4yznOWZvXrchnZDSjdHo?purpose=inline)

[datacamp.com](https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python?utm_source=chatgpt.com)

**B. Divisive Clustering (Top-Down Approach)**

Divisive Clustering works in the opposite direction of Agglomerative Clustering.

Steps:

1. Initially, all data points belong to one cluster.
2. Divide the cluster into smaller clusters.
3. Continue dividing clusters based on dissimilarities.
4. Stop when the desired number of clusters is obtained.

[Hierarchical Clustering: Build Trees That Reveal Hidden Data Structure - perfecXion.ai](https://images.openai.com/static-rsc-4/A4yQYZQt9B9eOSZsTKjssQ2JL_gXux_Tg5IPIknj-gCgXTMkBFJdFmKiDJnjaVjym7lDYAVfcBdFWFpQXMWVnoObCkntJwcuVsWt9QJpqsb55UJFp7mqCItx9ZvusyG5fsdUkYomsBBE_aDB2ium83aPQJ4drR9JpxFDiVDokAY?purpose=inline)

[perfecxion.ai](https://perfecxion.ai/articles/hierarchical-clustering-complete-guide.html?utm_source=chatgpt.com)

**7. What is a Dendrogram?**

A Dendrogram is a tree-like diagram used to represent the merging or splitting of clusters in Hierarchical Clustering.

It helps us understand how data points are grouped together.

[Unsupervised Clustering in Python with Scikit-learn](https://images.openai.com/static-rsc-4/4xYDQV3TPVwA0nwQCAyCBxxDOngxG8P_4FMXzfY_kg_Sbnlj6EQ9GpFODvUpjtXyTIdhwwZSFeqwfSBDirDAoPxNyakvC687nR5dWN6Ms7RKSP0AkABFxZE4E7low8YZGdqmJpRTKl1MfOsG4YWLPin3NdBBNf6rEJJNXVSeppo?purpose=inline)

[programming-partner.com](https://www.programming-partner.com/blog/unsupervised-learning-clustering-python-scikit-learn?utm_source=chatgpt.com)

**How to read a Dendrogram?**

- Bottom: Individual data points.
- Branches: Show how clusters are merged.
- Height: Represents the distance at which clusters are merged.
- Horizontal cut: Determines the number of clusters.

If we draw a horizontal line across the dendrogram, the number of branches it intersects indicates the number of clusters.

**Types of Linkage Methods**

Linkage determines how the distance between two clusters is calculated.

| Linkage          | Description                                                         |
| ---------------- | ------------------------------------------------------------------- |
| Single Linkage   | Minimum distance between two clusters                               |
| Complete Linkage | Maximum distance between two clusters                               |
| Average Linkage  | Average distance between all pairs of points                        |
| Ward Linkage     | Merges clusters to minimize the increase in within-cluster variance |

**Advantages of Hierarchical Clustering**

1. Does not require specifying K initially.
2. Provides a visual representation through dendrograms.
3. Helps understand relationships between data points.
4. Supports different distance and linkage methods.

**Disadvantages of Hierarchical Clustering**

1. Computationally expensive for large datasets.
2. Sensitive to noise and outliers.
3. Once clusters are merged, they generally cannot be separated again in agglomerative clustering.
4. Results depend on the selected distance and linkage method.

**8. Difference Between K-Means and Hierarchical Clustering**

| K-Means                                        | Hierarchical Clustering                      |
| ---------------------------------------------- | -------------------------------------------- |
| Partition-based clustering algorithm.          | Hierarchy-based clustering algorithm.        |
| Requires K initially.                          | Does not require K initially.                |
| Uses centroids.                                | Uses cluster distances and linkage.          |
| Uses repeated assignment and centroid updates. | Uses merging or splitting of clusters.       |
| Generally faster on large datasets.            | Generally more computationally expensive.    |
| No dendrogram is generated.                    | Generates a dendrogram.                      |
| Sensitive to initial centroid selection.       | Sensitive to distance and linkage selection. |

**9. Step-by-Step Python Implementation**

Now let's understand how to perform the actual assignment.

We will use Python and Scikit-learn.

**Step 1: Import Required Libraries**

```
import pandas as pdimport numpy as npimport matplotlib.pyplot as pltfrom sklearn.preprocessing import StandardScalerfrom sklearn.cluster import KMeansfrom sklearn.cluster import AgglomerativeClustering
```

Explanation:

| Library                 | Purpose                       |
| ----------------------- | ----------------------------- |
| Pandas                  | Reading and handling datasets |
| NumPy                   | Numerical operations          |
| Matplotlib              | Plotting graphs               |
| StandardScaler          | Feature normalization         |
| KMeans                  | K-Means clustering            |
| AgglomerativeClustering | Hierarchical clustering       |

**Step 2: Read the Dataset**

```
data = pd.read_csv("sales_data_sample.csv")print(data.head())print(data.shape)print(data.info())
```

- `read_csv()` loads the dataset.
- `head()` displays the first five rows.
- `shape` displays rows and columns.
- `info()` displays column information.

**Step 3: Select Numerical Features**

We need numerical features because K-Means calculates distances between data points.

For example, we can select:

- QUANTITYORDERED
- PRICEEACH
- SALES

```
X = data[    ['QUANTITYORDERED', 'PRICEEACH', 'SALES']]
```

We can also select other meaningful numerical sales features depending on the analysis.

**Step 4: Normalize the Data**

Normalization brings all numerical features into a similar scale.

We use StandardScaler.

```
scaler = StandardScaler()X_scaled = scaler.fit_transform(X)
```

Formula:

\\[ Z=\frac{X-\mu}{\sigma} \\]

Why is normalization important?

Suppose:

- Quantity Ordered = 50
- Price Each = 100
- Sales = 10,000

Sales has a much larger numerical range.

Without scaling, sales may dominate distance calculations.

StandardScaler prevents this problem by standardizing the features.

**Step 5: Apply the Elbow Method**

Now we determine the optimal number of clusters.

We test K values from 1 to 10 and calculate WCSS for each value.

```
wcss = []for i in range(1, 11):    kmeans = KMeans(        n_clusters=i,        init='k-means++',        random_state=42,        n_init=10    )    kmeans.fit(X_scaled)    wcss.append(kmeans.inertia_)
```

Explanation:

- `range(1, 11)`: Tests K values from 1 to 10.
- `n_clusters=i`: Specifies the number of clusters.
- `init='k-means++'`: Selects initial centroids using a smarter initialization method.
- `random_state=42`: Makes results reproducible.
- `n_init=10`: Runs K-Means with 10 different centroid initializations.
- `inertia_`: Returns WCSS.

**Step 6: Plot the Elbow Graph**

```
plt.plot(    range(1, 11),    wcss,    marker='o')plt.title("Elbow Method")plt.xlabel("Number of Clusters (K)")plt.ylabel("WCSS")plt.show()
```

The graph helps us identify the optimal K.

For example, suppose the elbow appears at K = 3.

Then we select three clusters for our final model.

**Step 7: Build the Final K-Means Model**

Suppose the selected optimal K is 3.

```
kmeans = KMeans(    n_clusters=3,    init='k-means++',    random_state=42,    n_init=10)data['Cluster'] = kmeans.fit_predict(X_scaled)print(data[['SALES', 'Cluster']].head())
```

Explanation:

- `n_clusters=3`: Creates three clusters.
- `fit_predict()`: Trains the model and assigns a cluster label to every data point.
- `data['Cluster']`: Stores the cluster number.

The output will contain cluster labels such as:

| Sales  | Cluster |
| ------ | ------- |
| 2,500  | 1       |
| 3,800  | 1       |
| 12,000 | 0       |
| 45,000 | 2       |
| 15,000 | 0       |

These values are illustrative. Actual cluster assignments depend on the dataset.

**Step 8: Visualize the Clusters**

We can visualize the clusters using a scatter plot.

```
plt.scatter(    X_scaled[:, 0],    X_scaled[:, 2],    c=data['Cluster'],    cmap='viridis')plt.title("K-Means Clustering")plt.xlabel("Quantity Ordered (Scaled)")plt.ylabel("Sales (Scaled)")plt.show()
```

Explanation:

- X-axis represents Quantity Ordered.
- Y-axis represents Sales.
- Different colors represent different clusters.

[How to Choose the Right Number of Clusters in the K-Means Algorithm? | by Manik Soni | The Startup | Medium](https://images.openai.com/static-rsc-4/MYvUypLw42YjlvZPm90p-YvFvTB-_LXEP_mQLNClLeRI436V6anTe01fBo67b8G-PSYBEmkycC8sZA_48IOXe-xXyFStjtdHGecW1-3Ns6niTpsQW9nPkGCiYQQDPsa66Mu97H0CVSB_QOfd3cFkf58PjT3RNuTNu_zZMIpATiQ?purpose=inline)

[medium.com](https://medium.com/swlh/how-to-choose-the-right-number-of-clusters-in-the-k-means-algorithm-9160c57ec760?utm_source=chatgpt.com)

**10. Hierarchical Clustering Implementation**

Now let's implement the second algorithm.

**Step 1: Import Libraries**

```
from scipy.cluster.hierarchy import dendrogram, linkagefrom sklearn.cluster import AgglomerativeClustering
```

**Step 2: Create a Dendrogram**

```
linked = linkage(    X_scaled,    method='ward')plt.figure(figsize=(10, 6))dendrogram(linked)plt.title("Hierarchical Clustering Dendrogram")plt.xlabel("Data Points")plt.ylabel("Distance")plt.show()
```

Explanation:

- `linkage()`: Calculates how clusters are merged.
- `method='ward'`: Uses Ward's linkage method.
- `dendrogram()`: Displays the hierarchical clustering tree.

The dendrogram helps us decide the number of clusters by examining large vertical gaps between merging distances.

**Step 3: Apply Agglomerative Clustering**

Suppose we decide to create three clusters.

```
hc = AgglomerativeClustering(    n_clusters=3,    linkage='ward')data['Hierarchical_Cluster'] = hc.fit_predict(X_scaled)print(data[['SALES', 'Hierarchical_Cluster']].head())
```

Explanation:

- `n_clusters=3`: Creates three groups.
- `linkage='ward'`: Minimizes the increase in within-cluster variance.
- `fit_predict()`: Fits the clustering model and assigns cluster labels.

**Step 4: Visualize Hierarchical Clusters**

```
plt.scatter(    X_scaled[:, 0],    X_scaled[:, 2],    c=data['Hierarchical_Cluster'],    cmap='rainbow')plt.title("Hierarchical Clustering")plt.xlabel("Quantity Ordered (Scaled)")plt.ylabel("Sales (Scaled)")plt.show()
```

This graph displays the data points belonging to different hierarchical clusters.

**11. How to Interpret the Final Clusters?**

After clustering, we should understand what each cluster represents.

For example, if we obtain three clusters, we might interpret them as:

| Cluster   | Possible Interpretation         |
| --------- | ------------------------------- |
| Cluster 0 | Low-value sales transactions    |
| Cluster 1 | Medium-value sales transactions |
| Cluster 2 | High-value sales transactions   |

However, these descriptions must be verified by calculating the actual average sales and other feature values in each cluster.

You can do this using:

```
print(    data.groupby('Cluster')[        ['QUANTITYORDERED', 'PRICEEACH', 'SALES']    ].mean())
```

This displays the average values of each feature for every cluster.

**12. Performance Evaluation of Clustering**

Unlike supervised learning, clustering does not have predefined correct labels in an ordinary unlabelled dataset.

Therefore, accuracy is generally not used to evaluate clustering.

We can use the following metrics:

| Metric               | Purpose                                    |
| -------------------- | ------------------------------------------ |
| WCSS / Inertia       | Measures within-cluster squared distances  |
| Silhouette Score     | Measures how well-separated clusters are   |
| Davies-Bouldin Index | Measures cluster similarity and separation |

**Silhouette Score**

It measures how similar a data point is to its own cluster compared with other clusters.

Its value ranges from -1 to +1.

- Close to +1: Good clustering.
- Around 0: Overlapping clusters.
- Close to -1: Data points may be assigned to the wrong cluster.

Python:

```
from sklearn.metrics import silhouette_scorescore = silhouette_score(X_scaled, data['Cluster'])print("Silhouette Score:", score)
```

A higher Silhouette Score generally indicates better-defined clusters.

**13. Important Viva Questions and Answers**

**Q1. What is clustering?**

Clustering is an unsupervised machine learning technique used to group similar data points into different clusters.

**Q2. What is K-Means?**

K-Means is an unsupervised clustering algorithm that divides data into K clusters based on the distance between data points and their centroids.

**Q3. What does K represent?**

K represents the number of clusters to be created.

**Q4. What is a centroid?**

A centroid is the mean position of all data points belonging to a particular cluster.

**Q5. What is WCSS?**

WCSS stands for Within-Cluster Sum of Squares. It measures the sum of squared distances between data points and their respective cluster centroids.

**Q6. What is the Elbow Method?**

The Elbow Method is a technique used to determine the optimal number of clusters by plotting WCSS against different values of K.

**Q7. Why is it called the Elbow Method?**

Because the graph looks like a human elbow at the point where the decrease in WCSS starts slowing down.

**Q8. What is Hierarchical Clustering?**

Hierarchical Clustering is an unsupervised learning algorithm that groups similar data points into a hierarchy of clusters.

**Q9. What is a dendrogram?**

A dendrogram is a tree-like diagram that represents the merging or splitting of clusters.

**Q10. What are the types of Hierarchical Clustering?**

- Agglomerative (Bottom-Up).
- Divisive (Top-Down).

**Q11. What is Agglomerative Clustering?**

It is a bottom-up approach in which every data point initially forms an individual cluster, and the closest clusters are repeatedly merged.

**Q12. What is Divisive Clustering?**

It is a top-down approach in which all data points initially belong to one cluster, which is repeatedly divided into smaller clusters.

**Q13. What is the difference between K-Means and Hierarchical Clustering?**

K-Means uses centroids and requires K beforehand, whereas Hierarchical Clustering creates a hierarchy of clusters using merging or splitting.

**Q14. Why is normalization required?**

Normalization brings features into a similar scale so that features with larger numerical values do not dominate distance calculations.

**Q15. Which distance formula is used in K-Means?**

Euclidean Distance is commonly used.

**Q16. What is the purpose of the random_state parameter?**

It ensures reproducible results by controlling the random initialization.

**Q17. Can K-Means handle categorical data directly?**

No. Categorical data must first be converted into a suitable numerical representation.

**Q18. What is the difference between supervised and unsupervised learning?**

Supervised learning uses labelled data, whereas unsupervised learning discovers patterns from unlabelled data.

**Q19. What is Silhouette Score?**

It measures how well a data point fits within its assigned cluster compared with other clusters.

**Q20. What happens if we select a very large K?**

The dataset may be divided into too many small clusters, potentially making the grouping less meaningful.

**14. Conclusion (Journal Answer)**

In this assignment, K-Means and Hierarchical Clustering algorithms are implemented on the `sales_data_sample.csv` dataset using Python.

The dataset is loaded and relevant numerical features are selected. The data is normalized using StandardScaler to bring all features into a similar scale.

The Elbow Method is used to determine the optimal number of clusters by calculating and plotting Within-Cluster Sum of Squares (WCSS) for different values of K.

K-Means clustering is then applied to group similar sales records into different clusters. Hierarchical Clustering is also implemented using Agglomerative Clustering, and a dendrogram is generated to visualize the cluster hierarchy.

Thus, clustering helps identify patterns in sales data and supports customer segmentation and business decision-making.

**15. Quick Revision: Understand These Concepts First**

| Priority | Concept                 | Remember                  |
| -------- | ----------------------- | ------------------------- |
| 1        | Clustering              | Grouping similar data     |
| 2        | Unsupervised Learning   | No predefined labels      |
| 3        | K-Means                 | Groups using centroids    |
| 4        | K                       | Number of clusters        |
| 5        | Centroid                | Center of a cluster       |
| 6        | Euclidean Distance      | Distance between points   |
| 7        | WCSS                    | Within-cluster variation  |
| 8        | Elbow Method            | Finds suitable K          |
| 9        | Hierarchical Clustering | Builds cluster hierarchy  |
| 10       | Dendrogram              | Tree-like cluster diagram |

**One important thing to remember for your practical:**

- K-Means → Elbow Method → WCSS → Optimal K.
- Hierarchical Clustering → Linkage → Dendrogram → Number of clusters.

If you understand these two flows, you will be able to explain almost the entire assignment in your viva.