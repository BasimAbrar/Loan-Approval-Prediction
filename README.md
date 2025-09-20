# Loan-Approval-Prediction
Loan approval prediction using classification models. Compared Logistic Regression and Decision Tree, with SMOTE and class weights for imbalance handling. Decision Tree achieved ~98% accuracy with balanced precision, recall, and F1-scores.
## Dataset  
- **Source:** Loan Approval Prediction Dataset (Kaggle)  
- **Description:** Contains applicant information such as income, education, loan amount, and employment status used to predict loan approval status.  

## Methodology  
1. **Data Preprocessing**  
   - Handled missing values.  
   - Encoded categorical features.  
   - Scaled numerical variables.  
2. **Model Training**  
   - Logistic Regression (baseline).  
   - Decision Tree (with and without class balancing).  
3. **Class Imbalance Handling**  
   - Applied SMOTE oversampling.  
   - Tested class weights in Decision Tree.  
4. **Evaluation Metrics**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  

## Results  
- **Logistic Regression**  
  - Accuracy: ~92%.  
  - Interpretable baseline but weaker performance on rejected loan cases.  
  - SMOTE had minimal impact on results.  

- **Decision Tree**  
  - Accuracy: ~98% (both balanced and unbalanced).  
  - Strong precision, recall, and F1-score across approved and rejected classes.  
  - Naturally handled the dataset’s mild imbalance.  

- **Final Model**  
  - **Decision Tree (without SMOTE or class weights)** → 98% accuracy with balanced performance across classes.  

## Tools & Libraries  
- Python  
- Pandas – data handling  
- Scikit-learn – Logistic Regression, Decision Tree, preprocessing, SMOTE  
- Matplotlib – visualization  
