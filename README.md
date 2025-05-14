# Multiclass Customer Sentiment Classification for E-Commerce Platform 

*Leveraged cross-functional collaboration and machine learning to improve customer experience by identifying and reducing negative customer feedback by over 30% through improved sentiment model retrainin and deployment.*

-----
![sentiment_analysis](https://github.com/user-attachments/assets/2c865ba1-89d4-41a6-a9f8-a15e8ae8b1ca)

# Executive Summary

MayFair is a fast-growing e-commerce startup in South Africa, recognized for its innovative strategies in digital commerce. The company aims to enhance its online shopping experience by improving customer engagement and optimizing workforce management through data-driven insights. To support these goals, the company seeks to leverage customer feedback data and predictive modeling for proactive decision-making.

## Project Overview

The project was divided into two teams:

* Team 1: Employee Attrition Modeling - Focused on developing a predictive model to identify employees at risk of leaving, helping HR implement timely interventions.

* Team 2: Customer Sentiment Analysis (Led by me) - Focused on automating the classification of customer sentiment from reviews to enhance customer engagement and feedback management.

## Business Objectives

**Customer Sentiment Analysis:**

* Automate sentiment classification of customer reviews.

* Integrate open-source reviews (e.g., AliExpress) to complement limited proprietary data.

* Prioritize and automate responses to negative feedback to improve customer satisfaction.

## Team 2 Project Details: Customer Sentiment Analysis

Project Duration: 4 Weeks

* Project Type: Data-Driven Business Insights

* Framework: Agile (Scrum-based)

* Team Members: Kehinde (Lead), Busayo, Florence, Felix

### Business Problem

MayFair aims to improve customer experience by leveraging sentiment analysis on customer feedback collected from e-commerce platforms, social media, and review sites. This data will help the company make data-driven decisions to optimize customer engagement and satisfaction.

### Methodology

1. **Data Collection:** We collaborated with the Data Engineering Team for data acquisition through web scraping and data extraction. The final dataset consists of 12,821 rows and 5 features:

  * Review ID : 

  * Review Content

  * Rating

  * Date

  * Country

The most critical feature for sentiment analysis is **Review Content**, which we used as the primary input for model training. The data contained 1% duplicate entries and 5% missing values in the review content field.

2. **Exploratory Data Analysis (EDA)**

 * **Word Cloud Analysis:**

![image](https://github.com/user-attachments/assets/110b5ef8-a6f1-495a-94f9-d606eccac6d2)


   * Positive reviews frequently contained words like 'great', 'good', 'easy' among other words.
   * Negative reviews often featured words like 'use', 'size', 'one', 'will' etc.

 The analysis revealed that no single word distinctly indicated a particular sentiment.

 * **Ratings Distribution**
   
The distribution of ratings shows that we have an imbalance dataset consist of 55% positive, 15% neutral and 30% negative reviews.

3. **Data Processing:** To ensure data quality, we performed the following preprocessing steps:

  * **Language Consistency:**
    
    * We ensured all reviews were in English by using a language detection function with a fixed seed for reproducibility. Non-English reviews were removed.

  * **Data Cleaning:**
    * Removed duplicate rows and entries with null reviews.
    * Removed emojis and URLs from the text.
    * Filtered out non-alphabetic characters.
    * Applied lemmatization to standardize word forms.
    * Removed common stopwords, while retaining negation terms to preserve sentiment context.

* **Negation Handling:**

   * Implemented a custom function to identify and tag words following negation terms (e.g., 'not good' → 'NOT_good') to enhance sentiment detection accuracy.

* After preprocessing, we obtained a clean dataset with the following sentiment distribution:

   ![image](https://github.com/user-attachments/assets/594a820c-63fb-4ac4-8980-96d8e13109bc)


   * Positive: 6,085 reviews
   * Negative: 3,411 reviews
   * Neutral: 1,757 reviews

4. **Feature Engineering and Data Splitting**

  * **Text Vectorization:**

    * Utilized TF-IDF (unigrams, bigrams, trigrams) to convert text into numerical features.
    * Selected the top 5,000 features using the chi-squared test to reduce dimensionality.

 * **Label Encoding:** Encoded sentiment labels (Positive, Negative, Neutral) using LabelEncoder.

 * **Handling Imbalanced Data:** Applied SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in the training data.

* **Mlflow:** Mlflow was used to track the models parameters and metrics

  
![image](https://github.com/user-attachments/assets/2edad0af-a010-484d-bdfe-aada3224c629)


5. **Modeling: Training and Evaluation **

 * **Rule-Based Models:** We established baseline performance using rule-based sentiment analysis techniques.
 * **ML Models:** And experimented with multiple machine learning models.

![image](https://github.com/user-attachments/assets/a054a4ac-1fc4-47ac-8144-21536984be76)


![image](https://github.com/user-attachments/assets/126cac18-783f-4814-a991-7de2b51d2f6b)


**Best Model: Support Vector Machine (SVM) with an accuracy of 0.78 and an AUC score of 0.86.**

**Why**:
  * Achieves the highest accuracy and AUC score.
  * Shows superior performance on the Positive class, which dominates the dataset.
  * Maintains a good balance between Positive and Negative predictions, despite challenges with the Neutral class.
  * Generalizes well to new data, as indicated by the AUC.

6. **Model Interpretation**

   * We used LIME (Local Interpretable Model-Agnostic Explanations) for interpretability, allowing us to explain individual predictions effectively.

7. **Deployment**

  * The best-performing model (SVM) was deployed using Streamlit, FastAPI, and Docker, and hosted on Render. The application can be accessed through the provided link.


# Conclusion

This project successfully classified customer reviews into positive, negative, and neutral sentiments using both rule-based and machine learning techniques. The final API deployment allows for real-time sentiment analysis, making it accessible for end users.





