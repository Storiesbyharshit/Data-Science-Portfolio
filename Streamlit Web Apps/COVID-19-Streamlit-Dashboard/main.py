import streamlit as st
import pandas as pd
import json
import urllib.request
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def mpl_plot(data, label, is_log):
	st.markdown("### "+label)
	ax = plt.axes()
	ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
	plt.bar(dates, data, color="red", label=label)
	if(is_log==True):
		plt.yscale("log")
	plt.xticks(rotation=60, fontsize=5)
	st.pyplot()

def overall_insights_window():
	st.markdown("## Overall Insights")
	# st.write("Total confirmed cases : %d \n\n Total recovered cases : %d \n\n Total deaths : %d" % (current_confirmed, current_recovered, current_deceased))
	st.line_chart(chart_data)

	st.markdown("### Closed cases stats")

	fig1 = plt.figure()
	ax1 = fig1.add_axes([0,0,1,1])
	ax1.axis('equal')
	category = ['Recovered', 'Dead']
	number = [current_recovered, current_deceased]
	ax1.pie(number, labels = category, autopct='%1.2f%%', radius=0.5)
	st.pyplot()

def detailed_charts_window():
	st.markdown("## Detailed Charts")
	daily_or_total = st.radio("Select what to plot",('Daily data', 'Cumulative (total) data'))

	options = ["Confirmed", "Recovered", "Deaths"]
	data_type = st.selectbox("Select what to plot", options, index=0, key=None)

	if daily_or_total=="Daily data":	
		if data_type=="Confirmed":
			mpl_plot(daily_confirmed, "Daily Confirmed", False)
		if data_type=="Recovered":
			mpl_plot(daily_recovered, "Daily Recovered", False)
		if data_type=="Deaths":
			mpl_plot(daily_deceased, "Daily Deaths", False)
	if daily_or_total=="Cumulative (total) data":
		log = st.checkbox("Logarithmic Scale", value=False)
		nature="Linear"
		if(log==True):
			nature="Logarithmic"
		if nature == "Linear":
			if data_type=="Confirmed":
				mpl_plot(total_confirmed, "Total Confirmed", False)
			if data_type=="Recovered":
				mpl_plot(total_recovered, "Total Recovered", False)
			if data_type=="Deaths":
				mpl_plot(total_deceased, "Total Deaths", False)

		if nature == "Logarithmic":
			if data_type=="Confirmed":
				mpl_plot(total_confirmed, "Total Confirmed", True)
			if data_type=="Recovered":
				mpl_plot(total_recovered, "Total Recovered", True)
			if data_type=="Deaths":
				mpl_plot(total_deceased, "Total Deaths", True)


def statewise_data_window():
	st.markdown("## Statewise Data")

	state = st.selectbox("Select state", statelist, index=0, key=None)
	info_dict = {}
	info_dict = stateinfo[state]
	st.markdown("## %s" % (state))
	st.markdown("Total Confirmed Cases : %s" % (info_dict["confirmed"]))
	st.markdown("Total Active Cases : %s" % (info_dict["active"]))
	st.markdown("Total Deaths : %s" % (info_dict["deaths"]))
	st.markdown("Total Recovered : %s" % (info_dict["recovered"]))

	st.markdown("## Compare amongst states")
	state_data_active = []
	state_data_confirmed = []
	state_data_deaths = []
	state_data_recovered = []
	state_code_list = []
	for s in statelist:
		state_data_active.append(int(stateinfo[s]["active"]))
		state_data_confirmed.append(int(stateinfo[s]["confirmed"]))
		state_data_deaths.append(int(stateinfo[s]["deaths"]))
		state_data_recovered.append(int(stateinfo[s]["recovered"]))
		state_code_list.append(stateinfo[s]["statecode"])

	comparelist = ["Active Cases", "Confirmed Cases", "Deaths", "Recovered Cases"]
	compare = st.selectbox("What to compare?", comparelist, index=0, key=None)
	if compare == "Active Cases":
		plt.bar(state_code_list, state_data_active, color="blue", label=compare)
		plt.xticks(rotation=90, fontsize=7, fontweight="bold")
		st.pyplot()
	if compare == "Confirmed Cases":
		plt.bar(state_code_list, state_data_confirmed, color="blue", label=compare)
		plt.xticks(rotation=90, fontsize=7, fontweight="bold")
		st.pyplot()
	if compare == "Deaths":
		plt.bar(state_code_list, state_data_deaths, color="blue", label=compare)
		plt.xticks(rotation=90, fontsize=7, fontweight="bold")
		st.pyplot()
	if compare == "Recovered Cases":
		plt.bar(state_code_list, state_data_recovered, color="blue", label=compare)
		plt.xticks(rotation=90, fontsize=7, fontweight="bold")
		st.pyplot()



st.title("COVID-19 India Dashboard")
st.sidebar.title("Coronavirus India Dashboard")
st.sidebar.markdown("India is one of the worst affected nations by the coronavirus outbreak.\
	This tool aims at providing realtime insights on the outbreak in India in the form of interactive charts.")

with urllib.request.urlopen("https://api.covid19india.org/data.json") as url:
    data = json.loads(url.read().decode())

daily_confirmed = np.zeros(np.size(data["cases_time_series"]))
daily_deceased = np.zeros(np.size(data["cases_time_series"]))
daily_recovered = np.zeros(np.size(data["cases_time_series"]))

total_confirmed = np.zeros(np.size(data["cases_time_series"]))
total_recovered = np.zeros(np.size(data["cases_time_series"]))
total_deceased = np.zeros(np.size(data["cases_time_series"]))


dates = []

i = 0
for d in data["cases_time_series"]:
	daily_confirmed[i] = d["dailyconfirmed"]
	daily_deceased[i] = d["dailydeceased"]
	daily_recovered[i] = d["dailyrecovered"]
	total_confirmed[i] = d["totalconfirmed"]
	total_recovered[i] = d["totalrecovered"]
	total_deceased[i] = d["totaldeceased"]

	dates.append(d["date"])
	i = i+1

current_confirmed = total_confirmed[np.size(total_confirmed)-1]
current_recovered = total_recovered[np.size(total_recovered)-1]
current_deceased = total_deceased[np.size(total_deceased)-1]

chart_data = np.transpose([daily_confirmed, daily_recovered, daily_deceased])
chart_data = pd.DataFrame(chart_data, columns=["Daily Confirmed Cases", "Daily Recovered Cases", "Daily Deaths"])

statelist = []
stateinfo = {}
for d in data["statewise"]:
	if(d["state"]!="Total"):
		statelist.append(d["state"])
		stateinfo[d["state"]] = d


st.sidebar.markdown("#### Total confirmed cases : %d " % (current_confirmed))
st.sidebar.markdown("#### Total recovered cases : %d " % (current_recovered))
st.sidebar.markdown("#### Total deaths : %d " % (current_deceased))

st.sidebar.markdown("\n")
window = st.sidebar.selectbox("Select:", ["Overall Insights", "Detailed Charts", "Statewise Data"], index=0, key=None)


if(window == "Overall Insights"):
	overall_insights_window()
if(window == "Detailed Charts"):
	detailed_charts_window()
if(window == "Statewise Data"):
	statewise_data_window()
