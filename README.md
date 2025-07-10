# Deep-Learning
🔍 Are you spending hours on EDA? There’s a smarter way!
As a data science & AI enthusiast, I know how important exploratory data analysis (EDA) is — but manual EDA can take hours if not done smartly.
This week, I explored the Heart Cleveland dataset (heart disease prediction) and used YData Profiling to turn hours of work into minutes!
📊 What is YData Profiling?
YData Profiling (previously pandas_profiling) is an amazing Python tool that automatically generates a detailed data report — including:
1-Missing values
2-Duplicates
3-Descriptive stats
4-Outliers
Correlation matrix (Pearson, Spearman, Kendall, Cramér’s V, etc.)
Visual distributions
…and much more! ⚡
💡 How did I use it?
I combined manual EDA + auto-profiling:
 ✅ Plotted numerical variable Age with histogram & KDE plot
 ✅ Checked its skewness
 ✅ Plotted categorical variable Sex with bar plots, count plots & pie chart
 ✅ Generated a complete HTML profiling report with just a few lines of code:
from ydata_profiling import ProfileReport
import pandas as pd
df = pd.read_csv('heart_cleveland_upload.csv')
profile = ProfileReport(df, title="Heart Dataset",
                        minimal=False,
                        correlations={
                            "pearson": {"calculate": True},
                            "spearman": {"calculate": True},
                            "kendall": {"calculate": True},
                            "phi_k": {"calculate": True},
                            "cramers": {"calculate": True},

                        })
profile.to_file('heart_profile_report.html')

✨ Why is this powerful?
✔️ Saves hours of EDA effort
 ✔️ Highlights data quality issues instantly
 ✔️ Helps you plan feature engineering smartly
 ✔️ Gives you confidence to build robust AI/ML models

🚀 My take:
 Before building any deep learning or ML model, I always make sure to understand my data inside out.
 YData Profiling is now my go-to tool for fast, detailed & reliable profiling.
📌 Have you tried YData Profiling yet?
 If you haven’t, I highly recommend it 
