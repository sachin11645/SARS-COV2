#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta 
from matplotlib import dates as mpl_dates
import datetime as dt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import folium


# In[3]:


df = pd.read_excel('COVID-19-geographic-disbtribution-worldwide.xlsx')
df


# In[4]:


df.columns.str.replace(r'\n','', regex=True)
df.columns = df.columns.str.replace(r'\n','', regex=True)
df.columns


# In[5]:


df.groupby('dateRep')['cases','deaths'].sum()


# In[6]:


df1 = df.groupby('countryterritoryCode')['cases','deaths'].sum()
df1


# In[7]:


df2 = df.groupby('countriesAndTerritories')['cases','deaths'].sum()
df2


# In[8]:


sorted_df2 = df2.sort_values('cases', ascending= False).reset_index()
sorted_df2


# In[9]:


sorted_df3 = df2.sort_values('deaths', ascending= False).reset_index()
sorted_df3


# In[10]:


sorted_df3 = df2.sort_values('deaths', ascending= False).reset_index()
sorted_df3


# In[11]:


sorted_d = df1.sort_values('cases', ascending= False).reset_index()
sorted_d


# In[12]:


ddlj = pd.merge(sorted_df2, sorted_d)
ddlj


# In[13]:


fig = go.Figure(data=go.Choropleth(
    locations = ddlj['countryterritoryCode'],
    z = ddlj['cases'],
    text = ddlj['countriesAndTerritories'],
  colorscale=[[0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'], [0.04, 'rgb(51,160,44)'],
            [0.05, 'rgb(178,223,138)'], [0.06, 'rgb(51,160,44)'],
            [0.07, 'rgb(251,154,153)'], [0.1, 'rgb(255,255,0)'],
            [0.13, 'rgb(251,154,153)'], [0.15, 'rgb(255,255,0)'],
            [0.16, 'rgb(178,223,138)'], [0.17, 'rgb(51,160,44)'],
            [0.18, 'rgb(251,154,153)'], [0.19, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']],    
    autocolorscale=False,
    reversescale=False,
    marker_line_color='black',
    marker_line_width=1,
    colorbar_title = 'Confirmed Cases',
))

fig.update_layout(
    title_text='Worldwide COVID19 Confirmed Cases',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0,
        y=-0.1,
        xref='paper',
        yref='paper',
        text='Dr. Sachin Subedi',
        showarrow = False
    )]
)


# In[14]:


fig = go.Figure(data=go.Choropleth(
    locations = ddlj['countryterritoryCode'],
    z = ddlj['cases'],
    text = ddlj['countriesAndTerritories'],
   colorscale = [[0,"rgb(5, 10, 172)"],[0.15,"rgb(40, 60, 190)"],[0.45,"rgb(70, 100, 245)"],\
                        [0.55,"rgb(90, 120, 245)"],[0.65,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
    autocolorscale=False,
    reversescale=True,
    marker_line_color='black',
    marker_line_width=0.5,
    colorbar_title = 'Confirmed cases',
))

fig.update_layout(
    title_text='Worldwide COVID19 cases',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0,
        y=-0.1,
        xref='paper',
        yref='paper',
        text='Dr. Sachin Subedi',
        showarrow = False
    )]
)

fig.show()


# In[15]:


fig = go.Figure(data=go.Choropleth(
    locations = ddlj['countryterritoryCode'],
    z = ddlj['deaths'],
    text = ddlj['countriesAndTerritories'],
  colorscale=[[0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'], [0.04, 'rgb(51,160,44)'],
            [0.05, 'rgb(178,223,138)'], [0.06, 'rgb(51,160,44)'],
            [0.07, 'rgb(251,154,153)'], [0.1, 'rgb(255,255,0)'],
            [0.13, 'rgb(251,154,153)'], [0.15, 'rgb(255,255,0)'],
            [0.16, 'rgb(178,223,138)'], [0.17, 'rgb(51,160,44)'],
            [0.18, 'rgb(251,154,153)'], [0.19, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']],    
    autocolorscale=False,
    reversescale=False,
    marker_line_color='black',
    marker_line_width=1,
    colorbar_title = 'Deaths',
))

fig.update_layout(
    title_text='Worldwide COVID19 Deaths',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0,
        y=-0.1,
        xref='paper',
        yref='paper',
        text='Dr. Sachin Subedi',
        showarrow = False
    )]
)

fig.show()


# In[16]:


sorted_dd2 = df2.sort_values('deaths', ascending= False).reset_index().head(10)
sorted_dd2


# In[17]:


fig = px.scatter(sorted_dd2,  x='countriesAndTerritories', y='deaths', size='deaths', 
                 color='countriesAndTerritories', hover_name='countriesAndTerritories', size_max=60)


fig.update_layout(title_text='Top 10 worst affected countries', template='plotly_dark'
            
)

fig.update_layout(
    annotations=[
        dict(
            x=-0.06,
            y=-0.15,
            showarrow=False,
            text="Dr. Sachin Subedi",
            xref="paper",
            yref="paper"
            )
            ,   
    ],

)

fig.show()


# In[18]:


fig = px.bar(
    sorted_dd2,
    x = "countriesAndTerritories",
    y = "deaths", 
    title= "Top 10 worst affected countries",
   color_discrete_sequence=["brown"], 
    height=550,
    width=850,
    text='deaths'
)
fig.update_traces(texttemplate='%{text:.2.5s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.update_layout(template='plotly_dark')

fig.update_layout(
    annotations=[
        dict(
            x=-0.1,
            y=-0.25,
            showarrow=False,
            text="Dr. Sachin Subedi",
            xref="paper",
            yref="paper",
            )
            ,   
    ],

)
fig.show()


# In[19]:


sorted_d2 = df2.sort_values('cases', ascending= False).reset_index().head(10)
sorted_d2


# In[20]:


fig = px.scatter(sorted_d2,  x='countriesAndTerritories', y='cases', size='cases', 
                 color='countriesAndTerritories', hover_name='countriesAndTerritories', size_max=60)
fig.update_layout(title_text='Top 10 worst affected countries'
            
)

fig.update_layout(
    annotations=[
        dict(
            x=-0.05,
            y=-0.15,
            showarrow=False,
            text="Dr. Sachin Subedi",
            xref="paper",
            yref="paper",
            )
            ,   
    ],

)

fig.show()


# In[21]:


fig = px.bar(
    sorted_d2,
    x = "countriesAndTerritories",
    y = "cases", 
    title= "Top 10 worst affected countries",
   color_discrete_sequence=["blue"], 
    height=550,
    width=850,
    text='cases'
)
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_layout(
    annotations=[
        dict(
            x=-0.1,
            y=-0.25,
            showarrow=False,
            text="Dr. Sachin Subedi",
            xref="paper",
            yref="paper",
            )
            ,   
    ],

)
fig.show()


# In[22]:


ts = df.groupby('dateRep')['cases','deaths'].sum().reset_index()
ts


# In[23]:


print(plt.style.available)


# In[24]:


plt.style.use('bmh')


# In[25]:


fig = px.bar(
    ts,
    x = "dateRep",
    y = "cases",
    title= "Daily worldwide new cases",
   color_discrete_sequence=["blue"], 
    height=500,
    width=800)

fig.show()


# In[26]:


fig = px.line(
    ts,
    x = "dateRep",
    y = "cases",
    title= "Daily worldwide new cases",
   color_discrete_sequence=["blue"], 
    height=500,
    width=800)

fig.update_layout(
    annotations=[
        dict(
            x=-0.1,
            y=-0.2,
            showarrow=False,
            text="Dr. Sachin Subedi",
            xref="paper",
            yref="paper",
            )
            ,   
    ],

)


# In[27]:


fig = px.bar(
    ts,
    x = "dateRep",
    y = "deaths",
    title= "Daily Worldwide deaths",
   color_discrete_sequence=["red"], 
    height=500,
    width=800)

fig.show()


# In[28]:


fig = px.line(
    ts,
    x = "dateRep",
    y = "deaths",
    title= "Daily Worldwide deaths",
   color_discrete_sequence=["red"], 
    height=500,
    width=800)

fig.update_layout(
    annotations=[
        dict(
            x=-0.1,
            y=-0.2,
            showarrow=False,
            text="Dr. Sachin Subedi",
            xref="paper",
            yref="paper",
            )
            ,   
    ],

)


# In[29]:


trace0 = go.Scatter(
x= ts.dateRep,
y= ts.cases,
mode = 'lines',
name = 'cases'
)

trace1 = go.Scatter(
x= ts.dateRep,
y= ts.deaths,
mode = 'lines',
name = 'deaths'
)

data = [trace0, trace1] 
layout = go.Layout(title = 'Daily Covid 19 scenario')
figure = go.Figure(data=data, layout=layout)

figure.update_layout(
    annotations=[
        dict(
            x=-0.07,
            y=-0.15,
            showarrow=False,
            text="Dr. Sachin Subedi",
            xref="paper",
            yref="paper",
           )
            ,   
    ],

)

figure.update_xaxes(title_text="Date")

figure.update_yaxes(title_text="Number of Casualties")


# In[30]:


ts['Cases'] = ts['cases'].cumsum()
ts['Deaths'] = ts['deaths'].cumsum()

trace0 = go.Scatter(
x= ts.dateRep,
y= ts.Cases,
mode = 'lines',
name = 'Cases'
)

trace1 = go.Scatter(
x= ts.dateRep,
y= ts.Deaths,
mode = 'lines',
name = 'Deaths'
)

data = [trace0, trace1] 
layout = go.Layout(title = 'Daily Covid 19 scenario')
figure = go.Figure(data=data, layout=layout)

figure.update_layout(
    annotations=[
        dict(
            x=-0.07,
            y=-0.15,
            showarrow=False,
            text="Dr. Sachin Subedi",
            xref="paper",
            yref="paper",
           )
            ,   
    ],

)

figure.update_xaxes(title_text="Date")

figure.update_yaxes(title_text="Number of Casualties")


# In[31]:


ts['cases_date'] = pd.to_datetime(ts['dateRep'])


# In[32]:


ts2 = ts.set_index('cases_date')
ts2


# In[33]:


plt.figure(figsize=(20,10))
sns.barplot(data=sorted_dd2,x='deaths', y='countriesAndTerritories', orient='h')
plt.xlabel("Deaths")
plt.xticks([i for i in range(5000, 180000, 10000)], rotation=90, fontsize=18)
plt.ylabel("Countries", fontsize=18)
plt.title('Worst affected nations', fontsize=20)


# In[34]:


NP = (df.loc[df['countriesAndTerritories'] == 'Nepal'])
NP1 = NP.groupby('dateRep')['cases','deaths'].sum().reset_index()
NP1


# In[35]:


NP2= NP1.set_index('dateRep')
NP2


# In[36]:


sns.set(style="white")
sns.relplot(x="dateRep", y="cases",  kind="line", height=5, 
    aspect=2,color='black', data=NP1)
plt.title('Daily cases in Nepal', fontsize=20)


# In[37]:


f, ax = plt.subplots(figsize = (12, 6))
ax.ticklabel_format(style='plain')
sns.set_color_codes('bright')
sns.barplot(x = 'cases', y = 'countriesAndTerritories', data = sorted_d2,
            label = 'cases', color = 'b', edgecolor = 'w')
sns.set_color_codes('muted')
sns.barplot(x = 'deaths', y = 'countriesAndTerritories', data = sorted_d2,
            label = 'deaths', color = 'r', edgecolor = 'w')
ax.legend(ncol = 2, loc = 'lower right')
sns.despine(left = True, bottom = True)
plt.title('Worst affected nations', fontsize=20)
plt.ylabel("Country", fontsize=18)
plt.xlabel("Number of Casualties", fontsize=18)
plt.show()


# In[38]:


f, axes = plt.subplots(1, 2, sharey=False, figsize=(15, 6))
sns.lineplot(x="dateRep", y="cases", color='blue', ax=axes[0], data=ts2)
sns.lineplot(x="dateRep", y="deaths", color='red', ax=axes[1], data=ts2)


# In[39]:


A = (df.loc[df['countriesAndTerritories'] == 'Nepal'])
A1 = A.groupby('dateRep')['cases','deaths'].sum()
A2 = A1.loc['2020-03-15':'2020-06-13']
A2 = A2.reset_index()


B = (df.loc[df['countriesAndTerritories'] == 'India'])
B1 = B.groupby('dateRep')['cases','deaths'].sum()
B2 = B1.loc['2020-03-15':'2020-06-13']
B2 = B2.reset_index()


C = (df.loc[df['countriesAndTerritories'] == 'Pakistan'])
C1 = C.groupby('dateRep')['cases','deaths'].sum()
C2 = C1.loc['2020-03-15':'2020-06-13']
C2 = C2.reset_index()


D = (df.loc[df['countriesAndTerritories'] == 'Afghanistan'])
D1 = D.groupby('dateRep')['cases','deaths'].sum()
D2 = D1.loc['2020-03-15':'2020-06-13']
D2 = D2.reset_index()


E = (df.loc[df['countriesAndTerritories'] == 'Sri_Lanka'])
E1 = E.groupby('dateRep')['cases','deaths'].sum()
E2 = E1.loc['2020-03-15':'2020-06-13']
E2 = E2.reset_index()


F = (df.loc[df['countriesAndTerritories'] == 'Bhutan'])
F1 = F.groupby('dateRep')['cases','deaths'].sum()
F2 = F1.loc['2020-03-15':'2020-06-13']
F2 = F2.reset_index()


G = (df.loc[df['countriesAndTerritories'] == 'Maldives'])
G1 = G.groupby('dateRep')['cases','deaths'].sum()
G2 = G1.loc['2020-03-15':'2020-06-13']
G2 = G2.reset_index()


H = (df.loc[df['countriesAndTerritories'] == 'Bangladesh'])
H1 = H.groupby('dateRep')['cases','deaths'].sum()
H2 = H1.loc['2020-03-15':'2020-06-13']
H2 = H2.reset_index()


# In[40]:


a = A.groupby('countriesAndTerritories')['cases','deaths'].sum().reset_index()
b = B.groupby('countriesAndTerritories')['cases','deaths'].sum().reset_index()
c = C.groupby('countriesAndTerritories')['cases','deaths'].sum().reset_index()
d = D.groupby('countriesAndTerritories')['cases','deaths'].sum().reset_index()
e = E.groupby('countriesAndTerritories')['cases','deaths'].sum().reset_index()
f = F.groupby('countriesAndTerritories')['cases','deaths'].sum().reset_index()
g = G.groupby('countriesAndTerritories')['cases','deaths'].sum().reset_index()
h = H.groupby('countriesAndTerritories')['cases','deaths'].sum().reset_index()


# In[41]:


ddlj =  a.append([b,c, d, e, f, g, h])
ddlj


# In[42]:


d1 = ddlj.sort_values('cases', ascending= False).reset_index().head(10)
d1


# In[43]:


sns.set(style="ticks")
f, ax = plt.subplots(figsize = (15, 8))
ax.ticklabel_format(style='plain')
sns.set_color_codes('bright')
sns.barplot(x = 'cases', y = 'countriesAndTerritories', data = d1,
            label = 'cases', color = 'b', edgecolor = 'w')
sns.set_color_codes('muted')
sns.barplot(x = 'deaths', y = 'countriesAndTerritories', data = d1,
            label = 'deaths', color = 'r', edgecolor = 'w')
ax.legend(ncol = 2, loc = 'lower right')
sns.despine(left = True, bottom = True)
ax.set_title("COVID19 in SAARC nations")
plt.ylabel("Country")
plt.xlabel("Number of Casualties")
plt.show()


# In[44]:


sns.set(style="ticks", rc={"lines.linewidth": 2})
f, axes = plt.subplots(1, sharey=False, figsize=(15, 8))
sns.lineplot(x="dateRep", y="cases", color='red', label="India", linestyle="-", data=B2)
sns.lineplot(x="dateRep", y="cases", color='black', label="Pakistan", linestyle="-",  data=C2)
sns.lineplot(x="dateRep", y="cases", color='green', label="Afghanistan", linestyle="-", data=D2)
sns.lineplot(x="dateRep", y="cases", color='yellow', label="Bangladesh", linestyle="-",  data=H2)
sns.lineplot(x="dateRep", y="cases", color='royalblue', label="Nepal", linestyle="-", data=A2)
axes.set_title("Daily COVID19 Cases in SAARC nations", fontsize=20)
sns.despine(left = True, bottom = True)
plt.legend(ncol = 1, loc = 'upper left')
plt.ylabel("Confirmed Cases", fontsize=18)
plt.xlabel("Date", fontsize=18)
plt.show()


# In[45]:


A = (df.loc[df['countriesAndTerritories'] == 'United_States_of_America'])
A1 = A.groupby('dateRep')['cases','deaths'].sum().reset_index()
A1['Cases'] = A1['cases'].cumsum()
A1['Deaths'] = A1['deaths'].cumsum()

sns.set(style="ticks", rc={"lines.linewidth": 3})

fig, ax1 = plt.subplots(figsize=(14,8))
ax2 = ax1.twinx()
ax1.ticklabel_format(style='plain')
sns.lineplot(data=A1, x="dateRep", y="Cases",  ax=ax1, label="Cases", color='blue')

sns.lineplot(data=A1, x="dateRep", y="Deaths", ax=ax2, label="Deaths", color='red')

ax1.set_xlabel("Date")
ax1.set_ylabel(r"Total Cases")
ax2.set_ylabel(r"Total Deaths")

ax1.legend(loc="upper left", bbox_to_anchor=(-0.07,0.93), bbox_transform=ax.transAxes)
ax2.legend(loc="upper left", bbox_to_anchor=(-0.07, 0.87), bbox_transform=ax.transAxes)

ax2.set_ylim(0, 350000)
ax.set_ylim(0,17000000)


plt.title("COVID19 in USA", fontsize=20)
plt.show()
sns.set()


# In[46]:


A = (df.loc[df['countriesAndTerritories'] == 'Nepal'])
A1 = A.groupby('dateRep')['cases','deaths'].sum().reset_index()
A1['Cases'] = A1['cases'].cumsum()
A1['Deaths'] = A1['deaths'].cumsum()

sns.set(style="ticks", rc={"lines.linewidth": 3})

fig, ax1 = plt.subplots(figsize=(15,6))
ax2 = ax1.twinx()

sns.lineplot(data=A1, x="dateRep", y="Cases",  ax=ax1, label="Cases", color='blue')

sns.lineplot(data=A1, x="dateRep", y="Deaths", ax=ax2, label="Deaths", color='red')

ax1.set_xlabel("Date", fontsize=18)
ax1.set_ylabel(r"Total Cases", fontsize=18)
ax2.set_ylabel(r"Total Deaths", fontsize=18)
ax2.set_ylim(0, 2100)
ax.set_ylim(0,25)

ax1.legend(loc="upper left", bbox_to_anchor=(-0.07,0.68), bbox_transform=ax.transAxes)
ax2.legend(loc="upper left", bbox_to_anchor=(-0.07,0.63), bbox_transform=ax.transAxes)

plt.title("COVID19 in Nepal", fontsize=20)
plt.show()
sns.set()


# In[47]:


A1 = ts.groupby('dateRep')['cases','deaths'].sum().reset_index()
A1['Cases'] = A1['cases'].cumsum()
A1['Deaths'] = A1['deaths'].cumsum()

f, axes = plt.subplots(1, 2, sharey=False, figsize=(15, 6))
sns.lineplot(x="dateRep", y="Cases", color='blue', ax=axes[0], data=A1)
sns.lineplot(x="dateRep", y="Deaths", color='red', ax=axes[1], data=A1)


# In[48]:


A1 = ts.groupby('dateRep')['cases','deaths'].sum().reset_index()
A1['Cases'] = A1['cases'].cumsum()
A1['Deaths'] = A1['deaths'].cumsum()
sns.set(style="ticks", rc={"lines.linewidth": 3})

fig, ax1 = plt.subplots(figsize=(15,7))
ax2 = ax1.twinx()
ax1.ticklabel_format(style='plain')
sns.lineplot(data=A1, x="dateRep", y="Cases",  ax=ax1, label="Cases", color='blue')

sns.lineplot(data=A1, x="dateRep", y="Deaths", ax=ax2, label="Deaths", color='red')

ax1.set_xlabel("Date", fontsize=18)
ax1.set_ylabel(r"Total Cases", fontsize=18)
ax2.set_ylabel(r"Total Deaths", fontsize=18)

ax2.set_ylim(0, 2000000)
ax.set_ylim(0,70000000)

ax1.legend(loc="upper left", bbox_to_anchor=(-0.06,0.8), bbox_transform=ax.transAxes)
ax2.legend(loc="upper left", bbox_to_anchor=(-0.06,0.75), bbox_transform=ax.transAxes)
plt.title("COVID19 in World", fontsize=20)
plt.show()
sns.set()


# In[49]:


A = (df.loc[df['countriesAndTerritories'] == 'China'])
A1 = A.groupby('dateRep')['cases','deaths'].sum().reset_index()
A1['Cases'] = A1['cases'].cumsum()
A1['Deaths'] = A1['deaths'].cumsum()

sns.set(style="ticks", rc={"lines.linewidth": 3})

fig, ax1 = plt.subplots(figsize=(15,6))
ax2 = ax1.twinx()

sns.lineplot(data=A1, x="dateRep", y="Cases",  ax=ax1, label="Cases", color='blue')

sns.lineplot(data=A1, x="dateRep", y="Deaths", ax=ax2, label="Deaths", color='red')

ax1.set_xlabel("Date", fontsize=18)
ax1.set_ylabel(r"Total Cases", fontsize=18)
ax2.set_ylabel(r"Total Deaths", fontsize=18)

ax2.set_ylim(0, 5000)
ax.set_ylim(0,90000)

ax1.legend(loc="upper left", bbox_to_anchor=(-0.07,0.68), bbox_transform=ax.transAxes)
ax2.legend(loc="upper left", bbox_to_anchor=(-0.07,0.63), bbox_transform=ax.transAxes)

plt.title("COVID19 in China", fontsize=20)
plt.show()
sns.set()


# In[53]:


NP = (df.loc[df['countriesAndTerritories'] == 'Nepal'])
train =  NP.groupby('dateRep')['cases','deaths'].sum().reset_index()
test =  NP.groupby('dateRep')['cases','deaths'].sum().reset_index()
train['Cases'] = train['cases'].cumsum()
test['Cases'] = test['cases'].cumsum()

train.rename(columns={"dateRep":"ds","Cases":"y"},inplace=True)
test.rename(columns={"dateRep":"ds","Cases":"y"},inplace=True)
test = test.set_index("ds")
test = test['y']

from statsmodels.tsa.arima_model import ARIMA
import datetime
arima = ARIMA(train['y'], order=(3, 1, 0))
arima = arima.fit(trend='nc', full_output=True, disp=True)
forecast = arima.forecast(steps= 30)
pred = list(forecast[0])
start_date = train['ds'].max()
prediction_dates = []
for i in range(30):
    date = start_date + datetime.timedelta(days=1)
    prediction_dates.append(date)
    start_date = date
plt.figure(figsize= (20,10))
plt.xlabel("Dates",fontsize = 10)
plt.ylabel('Total cases',fontsize = 10)
plt.title("Predicted Values for the next 25 Days" , fontsize = 20)

plt.plot_date(y= pred,x= prediction_dates,linestyle ='dashed',color = 'crimson',label = 'Predicted')
plt.plot_date(y=train['y'].tail(15),x=train['ds'].tail(15),linestyle = '-',color = 'blue',label = 'Actual')
plt.style.use('bmh')

