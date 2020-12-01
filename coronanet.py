import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from colour import Color

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

def get_data():
	"""
		gets 'live' data from github
	"""
	country="Germany"
	url="https://raw.githubusercontent.com/saudiwin/corona_tscs/master/data/CoronaNet/data_country/coronanet_release/coronanet_release_{0}.csv".format(country)
	data=pd.read_csv(url, encoding='iso-8859-1')
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
	data = data[~(data.target_province.isin(["Berlin Brandenburg", "Bayern Baden-Württemberg", "Gütersloh, Warendorf"]))]

	return data

def create_graph():
	data = get_data()
	data = clean_bundeslaender(data)

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

	#datasets
	started_within_last_2w = data[data.date_start > threshold_2w]
	ongoing = data[(data.date_start < threshold_2w) & (data.date_start > threshold_4w) & (data.date_end > threshold_2w)]
	ongoing_4w = data[(data.date_start < threshold_4w) & (data.date_end > threshold_2w)]
#	ended_within_2w = data[(data.date_start < threshold_4w & data.date_end > threshold_2w)] # TODO
#	ended_2w_4w = data[(data.date_start < threshold_4w & data.date_end > threshold_2w)] # TODO

	##############
	### PLOTLY ###
	##############

	### NODES
	nodes = []
	for i,node in enumerate(u_provinces):
		x = 0.35
		y = i
		r_value = R_VALUES[node]
		nodes.append({
			'key': node,
			'x': x,
			'y': y,
			'textpos': "middle left",
			'r_value': r_value,
			'color': get_color_for_r_value(r_value),
			'hovertext': 'R-Value: {0}'.format(r_value)
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
			'hovertext': 'R-Value: {0}'.format(r_value)
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
			size=30,
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
