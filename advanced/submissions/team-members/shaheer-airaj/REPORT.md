🩺 CardioSentinel – Advanced Track

✅ Week 1: Exploratory Data Analysis (EDA)

📦 1. Data Integrity & Structure  
Q: Are there any missing, duplicate, or inconsistent entries in the dataset?  
A: Nope, data is fairly consistent 

Q: Are all feature data types appropriate (e.g., numeric, categorical)?  
A: Some binary features were labeled as integer which should be actually categorical data

Q: Did you detect any irrelevant, constant, or low-variance features that might be removed?  
A: None 

🎯 2. Target Variable Assessment  
Q: What is the distribution of the Heart_Attack_Risk target variable (0 = Not at Risk, 1 = At Risk)?  
A: 64% class 0 and 35% class 1 

Q: Is there a class imbalance between the two target classes? If so, how severe is it?  
A: Imbalance is not severe. We can continue as is.

Q: How might this imbalance affect your model choice or evaluation metrics later on?  
A: No significant imbalance detected 

📊 3. Feature Distribution & Quality  
Q: Which numerical features (e.g., Age, Cholesterol, BloodPressure, HeartRate) are skewed or contain outliers?  
A: No outliers detected 

Q: Did any features contain unrealistic or problematic values that may require capping or removal?  
A: None 

Q: What transformation or normalization techniques might improve these distributions?  
A: Distributions are ideal so no transformations required

📈 4. Feature Relationships & Patterns  
Q: Which categorical features (e.g., Smoking, AlcoholConsumption, Diet, PhysicalActivity) show patterns in relation to heart-attack risk?  
A: None of them show any significant patterns. Upon analyzing the chi-square test and T-test results, it was concluded that no feature displays any siginifance to the target variable. 

Q: Are there any strong pairwise relationships or multicollinearity among clinical variables (e.g., Cholesterol, BloodPressure, BMI)?  
A: No multi-colinearity between features detected.

Q: What trends or correlations stood out most clearly during your EDA?  
A: What satnds out is that the data is clean and surprisingly well-balanced which makes me think that it is a curated dataset not particularly useful to conduct analysis.

🧰 5. EDA Summary & Preprocessing Plan  
Q: What are your 3–5 key insights from the EDA about heart-attack risk factors?  
A: No significant patterns to note as the dataset seems very sythetic

Q: Which features will you scale, encode, or exclude during preprocessing?  
A: Will have further look at blood pressure and get interaction features by using ratios


✅ Week 2: Feature Engineering & Preprocessing  

🏷️ 1. Feature Encoding  
Q: Identify binary categorical features (e.g., Smoking, AlcoholConsumption) and apply simple encoding. Which features did you encode?  
A:  

Q: Apply ordinal encoding to lifestyle variables with ranked categories (e.g., Diet quality or PhysicalActivity frequency). What order did you assign, and why?  
A:  

Q: For remaining nominal categorical features, apply one-hot encoding. Why is one-hot encoding preferable for non-ordinal categories?  
A:  

✨ 2. Feature Creation  
Q: Create a new feature `BMI_Category` (Underweight, Normal, Overweight, Obese) based on BMI ranges. Display its value counts.  
A:  

Q: Create a `Risk_Index` feature using the formula `(Cholesterol + BloodPressure) / ExerciseHours`. Explain the rationale behind this derived metric.  
A:  

Q: Did either of the engineered features show a visible relationship with heart-attack risk?  
A:  

✂️ 3. Data Splitting  
Q: Split your dataset into training and testing sets (80/20 recommended). Use stratification on the Heart_Attack_Risk target.  
A:  

Q: Why must the dataset be split before applying SMOTE or scaling techniques?  
A:  

Q: Show the shape of your X_train, X_test, y_train, and y_test arrays to confirm the split.  
A:  

⚖️ 4. Imbalance Handling & Final Preprocessing  
Q: Apply the SMOTE technique (or class weighting) on the training set to handle imbalance. Show the class distribution before and after resampling.  
A:  

Q: Normalize numerical features using StandardScaler (fit on training data only). Why must you not fit the scaler on the test set?  
A:  

Q: Display the final shape of your preprocessed training (X_train_processed) and testing (X_test_processed) feature matrices.  
A:  
