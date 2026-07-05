# 🛒 Customer Shopping Behavior Analysis

## 📌 Project Overview
This project analyzes a **Customer Shopping Behavior dataset** using Python.  
The goal is to understand customer purchase patterns, subscription behavior, and predict subscription status using machine learning models.  
The workflow includes **data preprocessing, visualization, feature engineering, and classification modeling**.

---

## ⚙️ Requirements
Install the following Python libraries before running the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm

Dataset Information
File: customer_shopping_behavior.csv

Rows: 10,000

Columns: 26

Key Features:

Age, Gender, Location

Online/Offline, Online Store

Category, Item Purchased, Brand, Color, Size

Purchase Amount (₹), Discount (%), Festival/Sale

Delivery Speed, Delivery Time (Days)

Subscription Status (Target variable: Yes/No)

Payment Method, Review Rating, Return Status

Previous Purchases, Frequency of Purchases

Date features: Month, Purchase_Year, Purchase_Month, Purchase_Day

🧹 Data Preprocessing
Checked for missing values → none found.

Converted Purchase Date to datetime and extracted month/year/day.

Filled missing values:

Size → replaced with mode by age & category.

Online Store → replaced with "Near_by_shop".

Delivery Speed → inferred from Delivery Time (Days) (Same Day, Express, Standard).

Mapped festival sales to months (e.g., January → Republic Day Sales).

Encoded categorical variables using LabelEncoder.

Scaled numeric features using StandardScaler.

📊 Exploratory Data Analysis
Histograms comparing subscribed vs. non-subscribed customers across numeric attributes.

Unique values explored for:

Size (L, M, Free Size, numeric shoe sizes)

Online Store (Flipkart, Myntra, Amazon, Ajio, Meesho, etc.)

Delivery Speed (Same Day, Express, Standard)

Festival/Sale (Republic Day, Diwali, Christmas, etc.)

🤖 Machine Learning Models
Several classification models were trained to predict Subscription Status:

Model	Accuracy	Key Notes
Logistic Regression	68%	Failed to predict class 1 (subscribers).
Naive Bayes	65%	Balanced but weaker precision for subscribers.
Random Forest Classifier	73%	Good recall for non-subscribers, weaker for subscribers.
Support Vector Classifier	68%	Similar to Logistic Regression, struggled with subscribers.
K-Nearest Neighbors (KNN)	67%	Moderate performance, weak recall for subscribers.
Decision Tree Classifier	69%	Balanced precision/recall, interpretable model.
Gradient Boosting (LightGBM)	81%	Strong performance, improved recall for subscribers.
LightGBM (Tuned with RandomizedSearchCV)	86%	Best model, balanced precision & recall across both classes.


📌 Key Insights
Subscription imbalance: More non-subscribers (≈ 5513) than subscribers (≈ 2487).

Festival/Sale mapping improved consistency in seasonal analysis.

Delivery Speed could be inferred from delivery days, reducing missing values.

LightGBM with hyperparameter tuning achieved the best accuracy (86%) and balanced performance across both classes.

🔮 Future Work
Apply SMOTE or class balancing to improve subscriber predictions.

Perform feature importance analysis to identify top drivers of subscription.

Build interactive dashboards for customer behavior visualization.

Extend to recommendation systems for personalized offers.

License
This project is for educational and research purposes only.
Dataset belongs to the original contributors.
