#!/usr/bin/env python
# coding: utf-8

# # Dog Shelter Visualizations
# ### Long Beach Animal Shelter Data 2020-2024
# 
# Using raw data from the Long Beach Animal Shelter (available at https://data.longbeach.gov/explore/dataset/animal-shelter-intakes-and-outcomes/information), I create a dataframe with the name, primary color, sex, DOB, age on intake (months), age category, intake condition, intake type, intake date, outcome type, outcome date, and time in shelter (months) for dogs that entered the shelter between 04/01/2020 to 03/30/2024.
# 
# The goal of the visualizations is to identify whether information available to the shelter on intake is predictive of the time dogs will spend in the shelter. To that end, the visualizations display interactions between attributes of the dogs (sex, intake type, and age) and time spent in the shelter.

# In[1]:


import pandas as pd
import altair as alt
import numpy as np

# Created a new dataframe for dogs

adopted_dogs = pd.read_csv(r"C:/Users/mmull/Documents/Python Scripts/animal-shelter-intakes-and-outcomes.csv")
mask = adopted_dogs["Outcome Type"] == 'RETURN TO OWNER'
adopted_dogs = adopted_dogs[~mask]

adopted_dogs = adopted_dogs.drop(columns=['Animal ID', 'Animal Type', 'Secondary Color', 'Intake Subtype', 'Reason for Intake', 'Crossing', 'Jurisdiction', 
                           'Outcome Subtype', 'latitude', 'longitude', 'intake_is_dead', 'outcome_is_dead', 'was_outcome_alive', 'geopoint'])
adopted_dogs['DOB'] = pd.to_datetime(adopted_dogs['DOB'])
adopted_dogs['Intake Date'] = pd.to_datetime(adopted_dogs['Intake Date'])
adopted_dogs['Outcome Date'] = pd.to_datetime(adopted_dogs['Outcome Date'])
adopted_dogs['Age on Intake'] = (adopted_dogs['Intake Date'] - adopted_dogs['DOB'])/np.timedelta64(1,'m')
adopted_dogs['Time in Shelter'] = (adopted_dogs['Outcome Date'] - adopted_dogs['Intake Date'])/np.timedelta64(1,'m')
mask2 = adopted_dogs['Age on Intake'] <= 0
adopted_dogs = adopted_dogs[~mask2]
bins = [0, 3, 12, 84, np.inf]
names = ['Puppy', 'Young', 'Adult', 'Senior']
adopted_dogs['Age Category'] = pd.cut(adopted_dogs['Age on Intake'], bins, labels = names, include_lowest = True)

adopted_dogs.head()


# ## Preliminary Visualizations
# 
# Below are the preliminary visualizations created to gather initial impressions of the data and narrow the focus of the final visualizations.
# 
# Both the violin plots and bar charts break the dataset down by age, sex, and intake type to identify patterns. Although the violin plots provide interesting information about the distribution within each category, I moved forward with the bar charts because they allow the color encoding seen in the final visualizations, which allows further exploration of the relationships between the categorical variables.
# 
# The scatterplot shows how long each dog spent in the shelter, which allows for analysis of overall patterns and the identification of outliers. In the final visualization, color encoding will be added to provide more information about the categorical variables.
# 
# Ultimately, the final visualizations focused on the relationship between age, intake reason, and length of shelter stay.

# In[39]:


violin1 = alt.Chart(adopted_dogs.dropna()).transform_density(
    'Time in Shelter',
    as_ = ['Time in Shelter', 'density'],
    extent = [0,8],
    groupby = ['Age Category']
).mark_area(orient='horizontal').encode(
    y = 'Time in Shelter',
    color = 'Age Category:N',
    x = alt.X(
    'density:Q',
    stack = 'center',
    impute = None,
    title = None,
    axis = alt.Axis(labels = False, values = [0], grid = False, ticks = True),
    ),
column = alt.Column(
    'Age Category:N',
    header = alt.Header(
        titleOrient = 'bottom',
        labelOrient = 'bottom',
        labelPadding = 0,
    ),
)
).properties(
width = 100
).configure_facet(
spacing = 0
).configure_view(
stroke = None)

violin2 = alt.Chart(adopted_dogs.dropna()).transform_density(
    'Time in Shelter',
    as_ = ['Time in Shelter', 'density'],
    extent = [0,8],
    groupby = ['Sex']
).mark_area(orient='horizontal').encode(
    y = 'Time in Shelter',
    color = 'Sex:N',
    x = alt.X(
    'density:Q',
    stack = 'center',
    impute = None,
    title = None,
    axis = alt.Axis(labels = False, values = [0], grid = False, ticks = True),
    ),
column = alt.Column(
    'Sex:N',
    header = alt.Header(
        titleOrient = 'bottom',
        labelOrient = 'bottom',
        labelPadding = 0,
    ),
)
).properties(
width = 100
).configure_facet(
spacing = 0
).configure_view(
stroke = None)

violin3 = alt.Chart(adopted_dogs.dropna()).transform_density(
    'Time in Shelter',
    as_ = ['Time in Shelter', 'density'],
    extent = [0,8],
    groupby = ['Intake Type']
).mark_area(orient='horizontal').encode(
    y = 'Time in Shelter',
    color = 'Intake Type:N',
    x = alt.X(
    'density:Q',
    stack = 'center',
    impute = None,
    title = None,
    axis = alt.Axis(labels = False, values = [0], grid = False, ticks = True),
    ),
column = alt.Column(
    'Intake Type:N',
    header = alt.Header(
        titleOrient = 'bottom',
        labelOrient = 'bottom',
        labelPadding = 0,
    ),
)
).properties(
width = 100
).configure_facet(
spacing = 0
).configure_view(
stroke = None)

violin1


# In[40]:


violin2


# In[41]:


violin3


# In[29]:


alt.Chart(adopted_dogs.dropna()).mark_bar().encode(
    x = alt.X(alt.repeat('column'), type = "nominal"),
    y = alt.Y('mean(Time in Shelter):Q', title = 'Mean Time in Shelter'),
).properties(
    width = 125,
    height = 125
).repeat(
    column = ['Age Category', 'Sex', 'Intake Type']
)


# In[44]:


alt.Chart(adopted_dogs.dropna()).mark_circle(size = 40).encode(
    x="Intake Date", 
    y = "Time in Shelter",
    tooltip = "Age Category").interactive(
    ).properties(
    title = "Time in Shelter")


# ## Final Visualizations
# 
# ### 1. Repeated Bar Chart
# 
# The following bar charts display the mean time in shelter for each category within the age category and intake type variables. The color encoding on the age category chart displays the proportion of each intake type within each age category. The color encoding on the intake type graph displays the proportion of each age category within each intake type. These charts allow for easy identification of which categories of each variable have higher or lower mean time in the shelter. It also allows some analysis of the relationship between age category and intake type.

# In[22]:


age = alt.Chart(adopted_dogs.dropna()).mark_bar().encode(
    x = "Age Category",
    y = alt.Y('mean(Time in Shelter):Q', title = 'Mean Time in Shelter (months)'),
    color = 'Intake Type'
).properties(
    width = 225,
    height = 225,
    title = 'Mean Time in Shelter by Age'
)
legend1 = alt.Chart(adopted_dogs.dropna()).mark_bar().encode(
    y = alt.Y('Intake Type', title = 'Intake Type'),
    color = 'Intake Type'
    )

intake = alt.Chart(adopted_dogs.dropna()).mark_bar().encode(
    x = "Intake Type",
    y = alt.Y('mean(Time in Shelter):Q', title = 'Mean Time in Shelter (months)'),
    color = alt.Color('Age Category', legend = None)
).properties(
    width = 225,
    height = 225,
    title = 'Mean Time in Shelter by Intake Type'
)
legend2 = alt.Chart(adopted_dogs.dropna()).mark_bar().encode(
    y = alt.Y('Age Category'),
    color = 'Age Category'
    )

age | legend1 | intake | legend2


# ### 2. Scatterplots of Intake and Outcome Dates
# 
# The scatterplots plots each dog's intake and outcome date. The scatterplot chart allows for analysis of the overall distribution of the data, including outliers. The color encoding on the first plot shows the intake type of each data point. The color encoding of the second plot shows the age category of each data point. By using selection to highlight each category, a user can visually compare the variablity of each category within the two variables. The user can also estimate the relative frequency of each category when it is highlighted.

# In[27]:


date_selection = alt.selection_point(fields=["Age Category"])
date_color = alt.condition(date_selection,
                    alt.Color("Age Category:N", legend = None),
                     alt.value("lightgray"))

date_scatter = alt.Chart(adopted_dogs.dropna()).mark_circle(size = 40).encode(
    x="Intake Date", 
    y = "Outcome Date",
    color = date_color,
    tooltip = "Age Category").interactive(
    ).properties(
    title = "Correlation between Intake and Outcome Date by Age")

date_legend = alt.Chart(adopted_dogs.dropna()).mark_circle(size = 40).encode(
    y = alt.Y("Age Category:N", axis = alt.Axis(orient="right")),
    color = date_color,
    ).add_params(
    date_selection)

selection = alt.selection_point(fields=["Intake Type"])
color = alt.condition(selection,
                    alt.Color("Intake Type:N", legend = None),
                     alt.value("lightgray"))

scatter = alt.Chart(adopted_dogs.dropna()).mark_circle(size = 60).encode(
    x="Intake Date", 
    y = "Outcome Date",
    color = color,
    tooltip = "Intake Type").interactive(
    ).properties(
    title = "Correlation between Intake and Outcome Date by Intake Type")

legend = alt.Chart(adopted_dogs.dropna()).mark_circle(size = 60).encode(
    y = alt.Y("Intake Type:N", axis = alt.Axis(orient="right")),
    color = color
    ).add_params(
    selection)

scatter | legend | date_scatter | date_legend


# # Evaluation
# 
# ## Procedure
# For a summative evaulation of the visualizations, I created a feedback form. I posted the visualizations and feedback form in an online data science forum to acquire particpants. The first question asked participants to use the visualizations to identify which profile of dog was most likely to spend a short time in the shelter. The following questions asked participants to identify which pair of graphs was easier to use, provided the most detail, and allowed for the identification of patterns in the data.
# 
# ## Findings
# Participants were able to accurately answer the first question, which required using the visualizations. The majority of participants found the box charts easier to use and preferable for identifying patterns in the data. Some participants responded that the scatter plots provided more detail. Overall, however, the box charts were the preferred visualization.
# 
# In the development of future visualizations, the inclusion of formative evaluations would be beneficial in order to strengthen the final product. 

# In[ ]:




