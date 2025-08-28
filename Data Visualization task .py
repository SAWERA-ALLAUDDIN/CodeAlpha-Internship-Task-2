import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")

# Load dataset
data = pd.read_csv(r"D:\Covid Data.CSV.csv")

# Survival column (Alive / Died)
data["SurvivalStatus"] = data["DATE_DIED"].apply(lambda x: "Alive" if x=="9999-99-99" else "Died")
data["SEX"] = data["SEX"].map({1: "Male", 2: "Female"})

# Age Groups
age_bins = [0,18,30,45,60,75,90,120]
age_labels = ["0-17","18-29","30-44","45-59","60-74","75-89","90+"]
data["AgeGroup"] = pd.cut(data["AGE"], bins=age_bins, labels=age_labels, right=False)

# Remove unknowns
data = data[~data["ICU"].isin([97,98,99])]
data = data[~data["INTUBED"].isin([97,98,99])]
data = data[~data["PREGNANT"].isin([97,98,99])]

# Map pregnant values to Yes/No for better readability
data["PREGNANT"] = data["PREGNANT"].map({1:"Yes", 2:"No"})

# Comorbidities
comorbidities = ["DIABETES","COPD","ASTHMA","INMSUPR","HIPERTENSION",
                 "CARDIOVASCULAR","OBESITY","TOBACCO"]

# Define palette
colors = ["#E4004B", "#ED775A", "#FAD691", "#C9CDCF"]

# ---------------- FIRST DASHBOARD ----------------
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))

# 1 Age Distribution
axes1[0,0].hist(data["AGE"], bins=40, edgecolor="black", color=colors[2])
axes1[0,0].set_title("Age Distribution", fontsize=12)

# 2 Survival vs Death Pie
data["SurvivalStatus"].value_counts().plot(kind="pie", autopct="%1.1f%%", 
                                           colors=[colors[0], colors[1]], ax=axes1[0,1])
axes1[0,1].set_ylabel("")
axes1[0,1].set_title("Survival vs Death", fontsize=12)

# 3 Gender vs Survival
pd.crosstab(data["SEX"], data["SurvivalStatus"]).plot(kind="bar", stacked=True, 
                                                      color=[colors[0], colors[1]], ax=axes1[1,0])
axes1[1,0].set_title("Gender vs Survival", fontsize=12)
axes1[1,0].set_ylabel("Count")

# 4 Deaths by Age Group
data[data["SurvivalStatus"]=="Died"]["AgeGroup"].value_counts().sort_index().plot(
    kind="bar", color=colors[0], ax=axes1[1,1])
axes1[1,1].set_title("Deaths by Age Group", fontsize=12)
axes1[1,1].set_ylabel("Number of Deaths")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("dashboard1.png", dpi=300)

# ---------------- SECOND DASHBOARD ----------------
# make figure taller to prevent overlap
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 14))

# 5 Comorbidity Correlation
sns.heatmap(data[comorbidities].corr(), annot=True, cmap="coolwarm", ax=axes2[0,0])
axes2[0,0].set_title("Comorbidity Correlation", fontsize=12)

# Rotate labels and shift upward
axes2[0,0].set_xticklabels(
    axes2[0,0].get_xticklabels(),
    rotation=45,
    ha="right",
    va="top"   # push labels upward
)
axes2[0,0].set_yticklabels(axes2[0,0].get_yticklabels(), rotation=0)

# 6 ICU vs Survival
pd.crosstab(data["ICU"], data["SurvivalStatus"]).plot(kind="bar", stacked=True, 
                                                      color=[colors[0], colors[1]], ax=axes2[0,1])
axes2[0,1].set_title("ICU vs Survival", fontsize=12)
axes2[0,1].set_ylabel("Count")

# 7 Pregnant vs Survival
pd.crosstab(data["PREGNANT"], data["SurvivalStatus"]).plot(kind="bar", stacked=True, 
                                                           color=[colors[0], colors[1]], ax=axes2[1,0])
axes2[1,0].set_title("Pregnant vs Survival (Females only)", fontsize=12)
axes2[1,0].set_ylabel("Count")
axes2[1,0].set_xlabel("Pregnant")

# 8 Intubation vs Survival
pd.crosstab(data["INTUBED"], data["SurvivalStatus"]).plot(kind="bar", stacked=True, 
                                                          color=[colors[0], colors[1]], ax=axes2[1,1])
axes2[1,1].set_title("Intubation vs Survival", fontsize=12)
axes2[1,1].set_ylabel("Count")

# add extra vertical space between rows
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3)
plt.savefig("dashboard2.png", dpi=300)

plt.show()



print("Shape of dataset:", data.shape)
print("\nData types:\n", data.dtypes)
print("\nSummary statistics:\n", data.describe())
print("\nMissing values:\n", data.isnull().sum())

print("Survival Status:\n", data["SurvivalStatus"].value_counts())
print("\nGender Distribution:\n", data["SEX"].value_counts())
print("\nAge Groups:\n", data["AgeGroup"].value_counts())

# --- Outlier Detection ---

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Boxplot for AGE
plt.figure(figsize=(8,5))
sns.boxplot(x=data["AGE"], color="#ED775A")
plt.title("Outlier Detection in AGE")
plt.xlabel("Age")
plt.show()

# 2. Boxplot for TOBACCO
plt.figure(figsize=(6,5))
sns.boxplot(x=data["TOBACCO"], color="#FAD691")
plt.title("Outlier Detection in TOBACCO")
plt.xlabel("Tobacco")
plt.show()

# 3. Check unusual values in TOBACCO
print("Unique values in TOBACCO:", data["TOBACCO"].unique())
print("Counts:\n", data["TOBACCO"].value_counts())

plt.figure(figsize=(10,6))
sns.heatmap(data[["DIABETES","COPD","ASTHMA","HIPERTENSION",
                 "CARDIOVASCULAR","OBESITY","TOBACCO"]].corr(), 
            annot=True, cmap="coolwarm")
plt.title("Correlation Between Comorbidities")
plt.show()

pd.crosstab(data["ICU"], data["SurvivalStatus"]).plot(kind="bar", stacked=True)
plt.title("ICU vs Survival")
plt.ylabel("Count")
plt.show()
