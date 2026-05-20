# Machine Learning Portfolio 🚀
This project was developed as a part of the JetBrains Academy (Hyperskill) curriculum to practice and implement machine learning algorithms.
## 1. Random Forest Classifier (from scratch)
**Dataset:** Titanic (survival prediction)

**Key Implementations:**
* **Bagging:** Implemented Bootstrap aggregation to train diverse trees.
* **Feature Subsampling:** Used `max_features='sqrt'` to ensure decorrelation between trees.
* **Majority Voting:** Custom logic to aggregate predictions from the forest.
* **Evaluation:** Optimized forest size by analyzing the accuracy curve.

## 2. Custom K-Means Clustering
**Dataset:** Wine (unsupervised grouping)

**Key Implementations:**
* **Centroid Optimization:** Iterative updates using Euclidean distance and convergence thresholds (`eps`).
* **Optimal K Selection:** Evaluated clustering quality using the **Elbow Method (Inertia)** and **Silhouette Score**.
* **Visualization:** Dimensionality reduction to compare predicted clusters with ground truth.

## 3. House Classification (Feature Engineering)
**Dataset:** House price categories

**Key Focus:** Comparative analysis of categorical encoding techniques.
* **One-Hot Encoding:** Applied to low-cardinality nominal data.
* **Ordinal Encoding:** Used for preserving inherent order in specific features.
* **Target Encoding:** Utilized the `category_encoders` library to handle high-cardinality features (Zip codes/Areas) by mapping them to target means.
* **Performance:** Evaluated models using Macro F1-Score to account for class imbalance.

## 4. Salary Prediction (Linear Regression)
**Dataset:** Salaries

**Key Implementations:**
* **Correlation Check:** Found highly correlated variables (Age and Experience) using a correlation matrix.
* **Feature Selection:** Used `itertools.combinations` to test which columns to drop to get the lowest MAPE score.
* **Fixing Predictions:** Handled unrealistic negative salary predictions by replacing them with zeros and medians.

---

## Technical Stack
* **Language:** Python 3.x
* **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Category Encoders, TQDM.

