prac 1 data collection cleaning modeling
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

risk = pd.read_csv('Saloni_Gavhane/risk_analytics_train.csv',index_col=0)

risk.head()

risk.dtypes

risk.shape

risk.columns

risk.isnull().sum()

risk.describe()
for x in['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term']:
  risk[x].fillna(risk[x].mode()[0])
  
  risk.isnull().sum()
  
  risk['LoanAmount'].fillna(round(risk['LoanAmount'].mean(),0))
  prac 3 perform statistics distribution
  import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats

mean = 78
std_dev = 25
total_students = 100
score = 60

#Calculate z-score for 60
z_score = (score - mean) / std_dev

#Calculate the probability of getting a score less than 60
prob = norm.cdf(z_score)

#Calculate the percentage of the student who go less than 60 marks
percent = prob * 100

#Print the result
print("Percentage of students who go less than 60 marks:", round(percent,2), "%")

mean = 78
std_dev = 25
total_students = 100
score = 70

#Calculate z-score for 60
z_score = (score - mean) / std_dev

#Calculate the probability of getting a score less than 60
prob = norm.cdf(z_score)

#Calculate the percentage of the student who go less than 70 marks
percent = (1-prob) * 100

#Print the result
print("Percentage of students who go less than 70 marks:", round(percent,2), "%")

from scipy.stats import binom

n = 6
p = 0.6

r_values = list(range(n + 1))

# list of pmf values
dist = [binom.pmf(r, n, p) for r in r_values]
print(dist)

plt.bar(r_values, dist)
plt.show()

stats.binom.pmf(5, 30, 0.15)

stats.poisson.cdf(10,20)

from scipy.stats import chi2

# Parameters for the Chi-Square distribution
degrees_of_freedom = 5  # Degrees of freedom

# Generate a range of x values
x = np.linspace(0, 20, 1000)


# Calculate the probability density function (PDF) for the Chi-Square distribution
pdf = chi2.pdf(x, degrees_of_freedom)

plt.plot(x, pdf, label=f'Chi-Square (df={degrees_of_freedom})')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()
plt.title('Chi-Square Distribution')
plt.grid()
plt.show()
prac 4 hypothesis testing
import numpy as np 
from scipy import stats

#Sample datu
group1 = np.array([85, 90, 88, 92, 95])
group2 = np.array([78, 80, 84, 83, 82])

#Perform Independent two-sample t-test 
f_stat, p_value = stats.f_oneway(group1, group2)

print(f"F-Statistic: {f_stat}") 
print(f"p-value: {p_value}")

#Interpolation
alpha = 0.05
if p_value < alpha:
    print("Null Hypothesis: There is significant difference between two groups.")
else:
    print("alternate Hypothesis: There is significant difference between two groups.")

    import numpy as np 
from scipy import stats

#Sample data
sample_heights = np.array([66, 68, 78, 65, 69])
population_mean = 67
population_std = 2 #Know poulation standard deviation

#calculate sample mean
sample_mean = np.mean(sample_heights)

#Perform one-sample z-test
z_stat = (sample_mean - population_mean)/(population_std/np.sqrt(len(sample_heights)))
p_value = (1 - stats.norm.cdf(np.abs(z_stat))) # two-tailed test

print(f"z-statistic: {z_stat}")
print("p-value:",p_value)

#Interpolation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The sample mean is significantly different from the population mean.")
else:
    print("Fail to reject the null hypothesis: The sample mean is not significantly different from the population mean.")
prac 5 categorical and binary data
df = pd.read_csv('Saloni_Gavhane/sales_data.csv')

df.head()

df.iloc[:,5:12].nunique()

df.describe()

category_counts = df['Country'].value_counts()

print(category_counts)

sns.countplot(data=df, x='Customer_Gender')
plt.title("Categorial Data Distribution")

plt.hist(df['Month'], bins=10, edgecolor='black')
plt.title('Revenue Histrogram')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.legend()

one_hot_encoded_data = pd.get_dummies(df, columns = ['Product'])
one_hot_encoded_data

one_hot_encoded_data = pd.get_dummies(df, columns = ['State'])
one_hot_encoded_data
prac 6 anova testing
import numpy as np

#Sample data: test scores for three different teaching methods
method1 = np.array([85, 88, 92, 94, 90])
method2 = np.array([79, 80, 83, 77, 79])
method3 = np.array([91, 89, 94, 96, 92])

from scipy import stats

#Perform one way ANOVA
f_stat, p_value = stats.f_oneway (method1,method2, method3)

print(f"F-Statistics: {f_stat}")
print(f"p_value: {p_value}")

import pandas as pd

#Sample data: test scores with two factors: teaching method and gender
data = pd.DataFrame({
    'Score': [85, 78, 91, 88, 88, 89, 92, 83, 94, 94, 77, 96, 90, 79, 92],
    'Method': ['Method1', 'Method1', 'Method1', 'Method2', 'Method2', 'Method2', 'Method3', 'Method3', 'Method3',
               'Method1', 'Method2', 'Method3', 'Method1', 'Method2', 'Method3'],
    'Gender': ['Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male',
             'Male', 'Male', 'Female', 'Female', 'Female']

    })

    import statsmodels.api as sm 
from statsmodels.formula.api import ols

#Fit the two-way ANOVA model with interaction
model = ols('Score ~ C(Method)*C(Gender)', data=data).fit()

#Perform two-way ANOVA (Type II sum of squares) 
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
prac 8 time series analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df['Volume'].plot()

df.plot(subplots = True , figsize =(6,6))

# Resampling the time series data based on monthly 'M' frequency 
df_month = df.resample("M").mean()

#using subplot 
fig, ax = plt.subplots(figsize=(6,6))

# ploting bar graph
ax.bar(df_month['2020':].index,
      df_month.loc['2020': , "Volume"],
      width = 20, align  = 'center')
prac 9 regression analysis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1,2,3,4,5])
y = np.array([7,14,15,18,19])
n = np.size(x)

x_mean = np.mean(x)
y_mean = np.mean(y)
x_mean, y_mean

Sxy = np.sum(x*y)- n*x_mean*y_mean
Sxx = np.sum(x*x)- n*x_mean*x_mean

b1 = Sxy/Sxx
b0 = y_mean - b1*x_mean
print('sloe b1 is', b1)
print('intercept b0 is', b0)

plt.scatter(x,y)
plt.xlabel("Independent variable X")
plt.ylabel('Dependent variable y')
plt.show()

y_pred = b1*x + b0
plt.scatter(x,y, color='red')
plt.plot(x,y_pred, color='green')
plt.xlabel('X')
plt.xlabel('Y')
plt.show()


  
