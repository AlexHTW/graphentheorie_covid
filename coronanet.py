# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from colour import Color
from pandas import DataFrame
import requests
import json
import random


THIN_EDGE = 0.3
THICK_EDGE = 3
MIN_R_VALUE = 0
MAX_R_VALUE = 3

N_COLORS = 3
COLORS = list(Color("green").range_to(Color("red"), N_COLORS))
COLORSCALE = [((i / N_COLORS), color.get_hex()) for i, color in enumerate(COLORS)]

R_VALUES = {
	'Baden-Wuerttemberg': random.uniform(0, 3),
	'Bremen': random.uniform(0, 3),
 	'Mecklenburg-Vorpommern': random.uniform(0, 3),
	'Bavaria': random.uniform(0, 3),
	'Lower Saxony': random.uniform(0, 3),
	'North Rhine-Westphalia': random.uniform(0, 3),
	'Saxony-Anhalt': random.uniform(0, 3),
	'Hamburg': random.uniform(0, 3),
	'Schleswig-Holstein': random.uniform(0, 3),
	'Hesse': random.uniform(0, 3),
	'Rheinland-Pfalz': random.uniform(0, 3),
	'Saarland': random.uniform(0, 3),
	'Saxony': random.uniform(0, 3),
	'Thuringia': random.uniform(0, 3),
	'Berlin': random.uniform(0, 3),
	'Brandenburg': random.uniform(0, 3),
	'Lombardy': random.uniform(0, 3)
}

R_VALUES['Countrywide'] = np.mean(list(R_VALUES.values()))


def get_cases_data_json():
	response = requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0/query?where=1%3D1&outFields=Bundesland,AnzahlFall,Meldedatum&outSR=800000000&f=json')
	data_json = json.loads(response.text)['features']
	
	data = pd.json_normalize(data_json, sep='_')
	data['attributes_Meldedatum'] = pd.to_datetime(data['attributes_Meldedatum'], unit='ms')
	#print(len(data))
	data = data.groupby(['attributes_Bundesland', 'attributes_Meldedatum'], as_index=False)['attributes_AnzahlFall'].sum()
	
	print(data.head())
	data = data.sort_values(['attributes_Bundesland', 'attributes_Meldedatum'])
	print(data)
	#print(get_unique_vals(data, 'attributes_Bundesland'))
	a =data[(data['attributes_Bundesland'] == 'Hamburg')]
	#plt.show()
	print(data.info())
	print(len(a))

def get_cases_data_csv():
	url = "https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv"
	data = pd.read_csv(url,
					 encoding= 'utf8', 
					 usecols=['Bundesland', 'AnzahlFall', 'Meldedatum'],
					 parse_dates=['Meldedatum'])
	
	bundesland_cases = data.groupby(['Bundesland', 'Meldedatum'], as_index = False)['AnzahlFall'].sum()
	bundesland_cases = bundesland_cases.set_index("Meldedatum").sort_index()
	#TODO: r-values calculation
	#bundesland_cases['r_value'] = bundesland_cases['AnzahlFall'].apply(lambda x: (bundesland_cases.iloc[-4:-1].sum(axis=1)) / (bundesland_cases.iloc[1:4].sum(axis=1)))
	#print(bundesland_cases.head())
	return bundesland_cases

def get_data():
	"""
		gets 'live' data from github
	"""
	country="Germany"
	url="https://raw.githubusercontent.com/saudiwin/corona_tscs/master/data/CoronaNet/data_country/coronanet_release/coronanet_release_{0}.csv".format(country)
	data=pd.read_csv(url, encoding='iso-8859-1')
	#data["target_province"] = data["target_province"].str.decode('iso-8859-1').str.encode('utf-8')
	return data

def get_unique_vals(data, col = "type"):
	"""
		returns unique values of a column
		call to (e.g.) data.type.unique()
	"""
	return getattr(getattr(data,col), 'unique')()

def get_color_for_r_value(r_value):
	"""
		for given r_value it returns corresponding color
	"""
	old_min = MIN_R_VALUE
	old_max = MAX_R_VALUE
	new_min = 0
	new_max = N_COLORS - 1
	idx = (((r_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
	return int(round(idx))
def get_size_for_number_of_cases(number_of_cases):
	"""
		for given number_of_cases it returns size of node
	"""

	return (int(round(number_of_cases / 1000)) +1)*5
def get_node_attr_by_key(nodes, key, attr):
	"""
		returns given attribute of element in nodes (=filtered by key)
	"""
	possible_nodes = [node for node in nodes if node['key']==key]
	if len(possible_nodes):
		return possible_nodes[0][attr]
	else:
		return None

def clean_bundeslaender(data):
	#data_all = data[(data.target_province.isin(['-', np.nan]))]
	data['target_province'] = data['target_province'].str.replace(';','')
	data['target_province'] = data['target_province'].str.replace(r'^-$','Countrywide')
	data[(data.target_province.isin(['-', np.nan]))] = data[(data.target_province.isin(['-', np.nan]))].assign(target_province = 'Countrywide')

	# MANUAL CLEANING
	# first Step: add missing rows
	for idx, row in data.iterrows():
		# seperate "Berlin Brandenburg" into each
		if row['target_province'] == "Berlin Brandenburg":
			row["target_province"] = "Berlin"
			data.append(row)
			row["target_province"] = "Brandenburg"
			data.append(row)
		# seperate "Berlin Brandenburg" into each
		elif row['target_province'] == "Bayern Baden-Württemberg":
			row["target_province"] = "Bavaria"
			data.append(row)
			row["target_province"] = "Baden-Wuerttemberg"
			data.append(row)
		elif row['target_province'] == "Gütersloh, Warendorf":
			row["target_province"] = "North Rhine-Westphalia"
			data.append(row)


	# second Step: remove rows
	# Data = All Data without having one of the values in brackets in the "taget_province" column
	# "~" is like a not, so the filter afterwards is reversed
	data = data[~(data.target_province.isin(["Berlin Brandenburg","Bayern Baden-Württemberg", "Gütersloh, Warendorf", "Lombardy"]))]

	return data

def clean_bundeslaender_cases(cases):
	cases = cases.replace(to_replace=r'^Baden-Württemberg', value='Baden-Wuerttemberg', regex=True)
	cases =cases.replace(to_replace=r'^Thüringen', value='Thuringia', regex=True)
	cases = cases.replace(to_replace=r'^Bayern', value='Bavaria', regex=True)
	cases = cases.replace(to_replace=r'^Niedersachsen', value='Lower Saxony', regex=True)
	cases = cases.replace(to_replace=r'^Sachsen', value='Saxony', regex=True)
	cases = cases.replace(to_replace=r'^Nordrhein-Westfalen', value='North Rhine-Westphalia', regex=True)
	cases = cases.replace(to_replace=r'^Hessen', value='Hesse', regex=True)
	return cases
def create_graph():
	#all data
	data = get_data()
	data = clean_bundeslaender(data) #provinces and measure
	
	#get provinces and measure types
	u_provinces = get_unique_vals(data, 'target_province')
	u_types = get_unique_vals(data, 'type')

	data['date_start'] = pd.to_datetime(data['date_start']).dt.date
	data['date_end'] = pd.to_datetime(data['date_end']).dt.date
	g = nx.Graph()
	g.add_nodes_from(u_provinces, bipartite=0)
	g.add_nodes_from(u_types, bipartite=1)
	#generate current types
	threshold_1w = date.today() - timedelta(days=7)
	threshold_2w = date.today() - timedelta(days=14)
	threshold_4w = date.today() - timedelta(days=28)

	started_within_last_2w = data[data.date_start > threshold_2w]
	ongoing = data[(data.date_start < threshold_2w) & (data.date_start > threshold_4w) & (data.date_end > threshold_2w)]
	ongoing_4w = data[(data.date_start < threshold_4w) & (data.date_end > threshold_2w)]
#	ended_within_2w = data[(data.date_start < threshold_4w & data.date_end > threshold_2w)] # TODO
#	ended_2w_4w = data[(data.date_start < threshold_4w & data.date_end > threshold_2w)] # TODO
	
	#get current cases of all province
	cases = get_cases_data_csv() #provinces and cases
	yesterday = date.today() - timedelta(days=2) #only the number of infections up to yesterday was recorded
	cases = cases.loc['{0}-{1}-{2}'.format(yesterday.year, yesterday.month, yesterday.day)]
	cases = clean_bundeslaender_cases(cases)
	cases = cases.reset_index().set_index("Bundesland")
	cases.loc['Countrywide']= cases.sum(numeric_only=True, axis=0)
	
	##############
	### PLOTLY ###
	##############

	### NODES
	nodes = []
	for i,node in enumerate(u_provinces):
		x = 0.35
		y = i
		r_value = R_VALUES[node]
		num_of_infec = cases.loc[node]['AnzahlFall']
		nodes.append({
			'key': node,
			'x': x,
			'y': y,
			'textpos': "middle left",
			'r_value': r_value,
			'color': get_color_for_r_value(r_value),
			'hovertext': 'R-Value: {0} , Number of cases: {1}'.format(r_value, num_of_infec), 
			'size': get_size_for_number_of_cases(num_of_infec)
		})

	for i,node in enumerate(u_types):
		x = 0.65
		y = i * (len(u_provinces) / len(u_types))
		nodes.append({
			'key': node,
			'x': x,
			'y': y,
			'textpos': "middle right",
			'color': "grey",
			'hovertext': 'R-Value: {0}'.format(r_value),
			'size': 30
		})

	node_trace = go.Scatter(
		x = [node['x'] for node in nodes],
		y = [node['y'] for node in nodes],
		text = [node['key'] for node in nodes], # Labels
		textposition = [node['textpos'] for node in nodes],
		mode='markers+text',
		hoverinfo='text',
		marker=dict(
			showscale=True,
			# colorscale options
			#'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
			#'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
			#'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
			colorscale=COLORSCALE,
			reversescale=True,
			color = [node['color'] for node in nodes],
			size= [node['size'] for node in nodes],
			colorbar=dict(
				thickness=15,
				title='R Value',
				xanchor='left',
				titleside='right'
			),
			line_width=2)
		)

	### EDGES
	edges = []

	# started within last 2w

	edge_x = []
	edge_y = []
	for idx, row in started_within_last_2w.iterrows():
		x0 = get_node_attr_by_key(nodes = nodes, key = row['target_province'], attr="x")
		y0 = get_node_attr_by_key(nodes = nodes, key = row['target_province'], attr="y")
		x1 = get_node_attr_by_key(nodes = nodes, key = row['type'], attr="x")
		y1 = get_node_attr_by_key(nodes = nodes, key = row['type'], attr="y")
		edge_x.append(x0)
		edge_x.append(x1)
		edge_x.append(None)
		edge_y.append(y0)
		edge_y.append(y1)
		edge_y.append(None)

	edges.append(go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=0.5, color='#888'),
		hoverinfo='none',
		mode='lines')
	)

	# ongoing

	edge_x = []
	edge_y = []
	for idx, row in ongoing.iterrows():
		x0 = get_node_attr_by_key(nodes = nodes, key = row['target_province'], attr="x")
		y0 = get_node_attr_by_key(nodes = nodes, key = row['target_province'], attr="y")
		x1 = get_node_attr_by_key(nodes = nodes, key = row['type'], attr="x")
		y1 = get_node_attr_by_key(nodes = nodes, key = row['type'], attr="y")
		edge_x.append(x0)
		edge_x.append(x1)
		edge_x.append(None)
		edge_y.append(y0)
		edge_y.append(y1)
		edge_y.append(None)

	edges.append(go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=0.5, color='#888'),
		hoverinfo='none',
		mode='lines')
	)

	# ongoing_4w

	edge_x = []
	edge_y = []
	for idx, row in ongoing_4w.iterrows():
		x0 = get_node_attr_by_key(nodes = nodes, key = row['target_province'], attr="x")
		y0 = get_node_attr_by_key(nodes = nodes, key = row['target_province'], attr="y")
		x1 = get_node_attr_by_key(nodes = nodes, key = row['type'], attr="x")
		y1 = get_node_attr_by_key(nodes = nodes, key = row['type'], attr="y")
		edge_x.append(x0)
		edge_x.append(x1)
		edge_x.append(None)
		edge_y.append(y0)
		edge_y.append(y1)
		edge_y.append(None)

	edges.append(go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=1, color='#888'),
		hoverinfo='none',
		mode='lines')
	)

	# node_adjacencies = []
	# node_text = []
	# for node, adjacencies in enumerate(g.adjacency()):
	#     node_adjacencies.append(len(adjacencies[1]))
	#     node_text.append('# of connections: '+str(len(adjacencies[1])))
	#
	# node_trace.marker.color = node_adjacencies
	# node_trace.text = node_text

	fig = go.Figure(data=[*edges, node_trace],
             layout=go.Layout(
                title='CoronaNet Visualization',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Data: <a href='https://www.coronanet-project.org/'> Coronanet Project</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
	fig.show()

if __name__ == '__main__':
	create_graph()
	