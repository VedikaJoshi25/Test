# =============================================================================
# EC2005 Data Analytics and Modelling Assignment
# This Notebook performs the following tasks:
# 1. Data Preparation: Filtering, renaming, and summarizing the dataset.
# 2. Exploratory Data Analysis (EDA): Descriptive statistics and visualizations.
# 3. Regression Analysis: Estimating five models.
# 4. Presentation: Organizing regression results and reporting goodness-of-fit.
#
# Note: All analyses are performed using a Lending Club dataset (10,000 loans)
#       contained in the file "loans_dataset.csv".
#
# =============================================================================

# ----------------------------
# Import Required Packages
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# If needed, install stargazer (uncomment below):
# !pip install stargazer

from stargazer.stargazer import Stargazer

# Set plot style for consistency
sns.set(style="whitegrid")

# ----------------------------
# Part A: Data Preparation
# ----------------------------

# Load the dataset
loans = pd.read_csv("loans_dataset.csv")
print("Number of observations in raw data:", len(loans))
print("Preview of raw data:")
print(loans.head())

# (a) Filter the dataset to retain only the required variables
variables_to_keep = [
    "interest rate", "verified_income", "debt_to_income", "total_credit_utilized", 
    "total_credit_limit", "public_record_bankrupt", "loan_purpose", "term",
    "inquiries_last_12m", "issue_month", "annual_income", "loan_amount", 
    "grade", "emp_length", "homeownership"
]
loans_cleaned = loans[variables_to_keep].copy()

# (b) Rename "inquiries_last_12m" to "credit_checks"
loans_cleaned = loans_cleaned.rename(columns={"inquiries_last_12m": "credit_checks"})

# (c) Report the number of observations before and after cleaning and summary stats
obs_before = len(loans)
obs_after = len(loans_cleaned)
print("\nObservations before cleaning:", obs_before)
print("Observations after cleaning:", obs_after)
print("Cleaned Data Columns:", loans_cleaned.columns.tolist())

print("\nSummary Statistics (Numerical Variables):")
print(loans_cleaned.describe())

print("\nSummary Statistics (All Variables):")
print(loans_cleaned.describe(include="all").round(2))

# ----------------------------
# Part B: Exploratory Data Analysis (EDA)
# ----------------------------

# 2.1 Descriptive Statistics

# (a) Descriptive statistics for selected numerical variables
selected_vars = ["interest rate", "annual_income", "debt_to_income", "loan_amount"]
summary_stats_selected = loans_cleaned[selected_vars].describe().round(2)
print("\nSummary Statistics for Selected Variables:")
print(summary_stats_selected)

# (b) Count unique values and frequencies for categorical variables

# For 'grade'
grade_counts = loans_cleaned["grade"].value_counts(dropna=False).sort_index().to_frame().T
print("\nGrade - Value Counts and Frequencies:")
print(grade_counts)

grade_stats = loans_cleaned["grade"].describe().round(2)
print("\nGrade - Summary Statistics:")
print(grade_stats.to_frame().T)

# For 'verified_income'
verified_income_counts = loans_cleaned["verified_income"].value_counts(dropna=False).sort_index().to_frame().T
print("\nVerified Income - Value Counts and Frequencies:")
print(verified_income_counts)
verified_income_stats = loans_cleaned["verified_income"].describe().round(2)
print("\nVerified Income - Summary Statistics:")
print(verified_income_stats.to_frame().T)

# For 'homeownership'
homeownership_counts = loans_cleaned["homeownership"].value_counts(dropna=False).sort_index().to_frame().T
print("\nHomeownership - Value Counts and Frequencies:")
print(homeownership_counts)
homeownership_stats = loans_cleaned["homeownership"].describe().round(2)
print("\nHomeownership - Summary Statistics:")
print(homeownership_stats.to_frame().T)

# 2.2 Visualizations

## (a) Histograms

# Histogram for interest rate
plt.figure(figsize=(8, 6))
sns.histplot(data=loans_cleaned, x="interest rate", kde=True, bins=44, color="darkblue")
plt.title("Distribution of Interest Rate")
plt.xlabel("Interest Rate (%)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("Distribution_of_Interest_Rate.png")
plt.show()

# Histogram for annual income
plt.figure(figsize=(8, 6))
sns.histplot(loans_cleaned["annual_income"], kde=True, bins=44, color="green")
plt.title("Distribution of Annual Income")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("Distribution_of_Annual_Income.png")
plt.show()

## (b) Scatterplots

# Scatterplot: Interest rate vs. Annual Income
plt.figure(figsize=(8, 6))
sns.scatterplot(data=loans_cleaned, x="annual_income", y="interest rate", 
                hue="annual_income", palette="coolwarm", alpha=0.6, s=40)
plt.title("Interest Rate vs. Annual Income")
plt.xlabel("Annual Income (£)")
plt.ylabel("Interest Rate (%)")
plt.xticks(ticks=[0, 500000, 1000000, 1500000, 2000000, 2300000])
plt.yticks(ticks=[0, 5, 10, 15, 20, 25, 30])
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Interest_Rate_vs_Annual_Income.png")
plt.show()

# Scatterplot with regression line: Interest rate vs. Debt-to-Income
sns.lmplot(data=loans_cleaned, x="debt_to_income", y="interest rate", height=6, aspect=1.33,
           scatter_kws={"color": "navy", "alpha":0.4, "s":30},
           line_kws={"color": "red", "linewidth":2}, ci=65)
plt.title("Interest Rate vs. Debt-to-Income Ratio")
plt.xlabel("Debt-to-Income Ratio (%)")
plt.ylabel("Interest Rate (%)")
plt.xticks(ticks=range(0, 500, 50))
plt.yticks(ticks=range(0, 35, 5))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Interest_Rate_vs_Debt_to_Income.png")
plt.show()

## (c) Boxplots

# Boxplot: Interest rate by Grade
plt.figure(figsize=(8, 6))
grade_order = ["A", "B", "C", "D", "E", "F", "G"]
sns.boxplot(data=loans_cleaned, x="grade", y="interest rate", color="skyblue", width=0.6, order=grade_order)
sns.despine()
plt.xlabel("Loan Grade", fontsize=14, fontweight="bold")
plt.ylabel("Interest Rate (%)", fontsize=14, fontweight="bold")
plt.title("Interest Rate by Grade", fontsize=16, fontweight="bold")
plt.ylim(0, 35)
plt.yticks(np.arange(0, 36, 5))
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("Interest_Rate_by_Grade.png")
plt.show()

# Boxplot: Interest rate by Verified Income
plt.figure(figsize=(8, 6))
sns.boxplot(data=loans_cleaned, x="verified_income", y="interest rate", width=0.6)
sns.despine()
plt.xlabel("Verified Income Status")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate by Verified Income")
plt.ylim(0, 35)
plt.yticks(np.arange(0, 36, 5))
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("Interest_Rate_by_Verified_Income.png")
plt.show()

# Boxplot: Interest rate by Homeownership
plt.figure(figsize=(8, 6))
sns.boxplot(data=loans_cleaned, x="homeownership", y="interest rate", color="yellow", width=0.6)
sns.despine()
plt.xlabel("Homeownership")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate by Homeownership")
plt.ylim(0, 35)
plt.yticks(np.arange(0, 36, 5))
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("Interest_Rate_by_Homeownership.png")
plt.show()

# 2.3 Derived Variables

# (a) Create and analyze credit_util = total_credit_utilized / total_credit_limit
# Set to 0 if total_credit_limit == 0.
loans_cleaned.loc[loans_cleaned["total_credit_limit"] != 0, "credit_util"] = (
    loans_cleaned.loc[loans_cleaned["total_credit_limit"] != 0, "total_credit_utilized"] /
    loans_cleaned.loc[loans_cleaned["total_credit_limit"] != 0, "total_credit_limit"]
)
loans_cleaned.loc[loans_cleaned["total_credit_limit"] == 0, "credit_util"] = 0

mean_credit_util = loans_cleaned["credit_util"].mean()
prop_non_zero_credit_util = (loans_cleaned["credit_util"] > 0).mean()
credit_util_stats = pd.DataFrame({
    "Mean": [round(mean_credit_util, 4)],
    "Proportion": [round(prop_non_zero_credit_util, 4)]
})
print("\nStatistics for Credit Utility:")
print(credit_util_stats)

# (b) Create bankruptcy_dummy: 0 if public_record_bankrupt = 0, else 1 if ≥1
loans_cleaned["bankruptcy_dummy"] = (loans_cleaned["public_record_bankrupt"] >= 1).astype(int)
mean_bankruptcy = loans_cleaned["bankruptcy_dummy"].mean()
prop_non_zero_bankruptcy = (loans_cleaned["bankruptcy_dummy"] > 0).mean()
bankruptcy_stats = pd.DataFrame({
    "Mean": [round(mean_bankruptcy, 4)],
    "Proportion": [round(prop_non_zero_bankruptcy, 4)]
})
print("\nStatistics for Bankruptcy Dummy:")
print(bankruptcy_stats)

# ----------------------------
# Part C: Regression Analysis
# ----------------------------

# a) Model 1: Simple Linear Regression (interest rate ~ debt_to_income)
model_1 = smf.ols("`interest rate` ~ debt_to_income", data=loans_cleaned).fit()
print("\nRegression Summary - Model 1:")
print(model_1.summary())
stargazer1 = Stargazer([model_1])
with open("regression_summary_model1.html", "w") as f:
    f.write(stargazer1.render_html())
print("\nEstimated Regression Equation (Model 1):")
print(f"Interest Rate = {model_1.params['Intercept']:.2f} + {model_1.params['debt_to_income']:.4f} * Debt-to-Income")
print("Interpretation: A one-unit increase in debt-to-income is associated with a change of " +
      f"{model_1.params['debt_to_income']:.4f} units in interest rate on average.")
p_value_beta1 = model_1.pvalues["debt_to_income"]
alpha_levels = [0.01, 0.05, 0.1]
for alpha in alpha_levels:
    if p_value_beta1 < alpha:
        print(f"At significance level {alpha}: Reject Ho: B1 = 0")
    else:
        print(f"At significance level {alpha}: Fail to reject Ho: B1 = 0")
print("Model 1 Coefficients:")
print(model_1.params)

# b) Model 2: Simple Linear Regression (interest rate ~ bankruptcy_dummy)
model_2 = smf.ols("`interest rate` ~ bankruptcy_dummy", data=loans_cleaned).fit()
print("\nRegression Summary - Model 2:")
print(model_2.summary())
stargazer2 = Stargazer([model_2])
with open("regression_summary_model2.html", "w") as f:
    f.write(stargazer2.render_html())
print("\nEstimated Regression Equation (Model 2):")
print(f"Interest Rate = {model_2.params['Intercept']:.2f} + {model_2.params['bankruptcy_dummy']:.4f} * bankruptcy_dummy")
print("Interpretation: A one-unit increase in bankruptcy_dummy is associated with a change of " +
      f"{model_2.params['bankruptcy_dummy']:.4f} units in interest rate on average.")
p_value_beta2 = model_2.pvalues["bankruptcy_dummy"]
for alpha in alpha_levels:
    if p_value_beta2 < alpha:
        print(f"At significance level {alpha}: Reject Ho: B1 = 0")
    else:
        print(f"At significance level {alpha}: Fail to reject Ho: B1 = 0")
print("Model 2 Coefficients:")
print(model_2.params)

# c) Model 3: Categorical Variable Regression (interest rate ~ verified_income dummies)
# Create dummies for verified_income with "Not Verified" as reference
loans_cleaned["verified_yes"] = (loans_cleaned["verified_income"] == "Verified").astype(int)
loans_cleaned["source_verified_yes"] = (loans_cleaned["verified_income"] == "Source Verified").astype(int)
model_3 = smf.ols("`interest rate` ~ verified_yes + source_verified_yes", data=loans_cleaned).fit()
print("\nRegression Summary - Model 3:")
print(model_3.summary())
stargazer3 = Stargazer([model_3])
with open("model3_regression_table.html", "w") as f:
    f.write(stargazer3.render_html())
print("\nAverage interest rate for unverified borrowers (Model 3):", model_3.params["Intercept"])
print("Coefficient for Verified (vs. 'Not Verified'):", model_3.params["verified_yes"])
print("Coefficient for Source Verified (vs. 'Not Verified'):", model_3.params["source_verified_yes"])

# d) Model 4: Multiple Regression (interest rate ~ debt_to_income + credit_util + bankruptcy_dummy)
required_columns = ["interest rate", "debt_to_income", "credit_util", "bankruptcy_dummy"]
all_present = all(col in loans_cleaned.columns for col in required_columns)
print("\nAll required columns present for Model 4:", all_present)
model_4 = smf.ols("`interest rate` ~ debt_to_income + credit_util + bankruptcy_dummy", data=loans_cleaned).fit()
print("\nRegression Summary - Model 4:")
print(model_4.summary())
stargazer4 = Stargazer([model_4])
with open("model4_regression_table.html", "w") as f:
    f.write(stargazer4.render_html())
print("\nInterpretation of Coefficients (Model 4):")
print("Intercept:", model_4.params['Intercept'])
print("Debt-to-Income:", model_4.params['debt_to_income'])
print("Credit Utilization:", model_4.params['credit_util'])
print("Bankruptcy Dummy:", model_4.params['bankruptcy_dummy'])

# e) Model 5: Enhanced Multiple Regression
model_5_vars = ["interest rate", "debt_to_income", "credit_util", "bankruptcy_dummy", 
                "annual_income", "loan_amount", "term", "grade", "emp_length", 
                "homeownership", "loan_purpose", "credit_checks"]
loans_model5 = loans_cleaned[model_5_vars].copy()
# Create dummy variables for categorical features (k-1 dummies)
loans_model5_dummies = pd.get_dummies(loans_model5, drop_first=True)
formula = "interest rate ~ " + " + ".join(loans_model5_dummies.columns.drop("interest rate"))
model_5 = smf.ols(formula, data=loans_model5_dummies).fit()
print("\nRegression Summary - Model 5:")
print(model_5.summary())
stargazer5 = Stargazer([model_5])
with open("model5_regression_table.html", "w") as f:
    f.write(stargazer5.render_html())
print("\nReference Categories (Model 5):")
print("term:", loans_cleaned["term"].unique()[0])
print("grade:", loans_cleaned["grade"].unique()[0])
print("emp_length:", loans_cleaned["emp_length"].unique()[0])
print("homeownership:", loans_cleaned["homeownership"].unique()[0])
print("loan_purpose:", loans_cleaned["loan_purpose"].unique()[0])
print("\nResiduals for the first 5 observations (Model 5):")
print(model_5.resid.head(5))

# ----------------------------
# Part D: Presentation and Discussion
# ----------------------------

from statsmodels.iolib.summary2 import summary_col

# Organize the regression results for Models 1 to 5
models = [model_1, model_2, model_3, model_4, model_5]
print("\nRegression Coefficients for Models 1 to 5:\n")
regression_table = summary_col(models, stars=True, float_format='%0.3f',
                               model_names=["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"])
print(regression_table)

# Goodness-of-Fit statistics table
gof_stats = pd.DataFrame({
    'Model': ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"],
    'R-squared': [round(m.rsquared, 3) for m in models],
    'Sample Size (N)': [int(m.nobs) for m in models],
    'F-statistic': [round(m.fvalue, 2) for m in models]
})
print("\nGoodness-of-Fit Statistics:")
print(gof_stats)
