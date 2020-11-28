import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime, date, timedelta
import plotly.graph_objects as go


THIN_EDGE = 0.3
THICK_EDGE = 3

def get_data():
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

def create_graph():
	data = get_data()
	#data_all = data[(data.target_province.isin(['-', np.nan]))]
	#data = data[~(data.target_province.isin(['-', np.nan]))]
	u_provinces = get_unique_vals(data, 'target_province')
	u_types = get_unique_vals(data, 'type')
	#append nanrows to data
	#for province in u_provinces:
	#	data = data.append(data_all.assign(target_province=province))

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

	#add edges
	g.add_weighted_edges_from(
		[(row['target_province'], row['type'], 0.2) for idx, row in started_within_last_2w.iterrows()],
		weight='weight',
		attr={'width': THIN_EDGE,'color': 'black', 'style': 'dashed'}
	)

	g.add_weighted_edges_from(
                [(row['target_province'], row['type'], 0.2) for idx, row in ongoing.iterrows()],
                weight='weight',
                attr={'width': THIN_EDGE,'color': 'black', 'style': 'solid'}
        )

	g.add_weighted_edges_from(
                [(row['target_province'], row['type'], 0.2) for idx, row in ongoing_4w.iterrows()],
                weight='weight',
                attr={'width': THICK_EDGE,'color': 'black', 'style': 'solid'}
        )

	pos = {node:[0.35, i] for i,node in enumerate(u_provinces)}
	pos.update({node:[0.65, i] for i,node in enumerate(u_types)})
	nx.draw_networkx_nodes(g, pos, node_size=50, node_color="red")
	nx.draw_networkx_edges(
				g,
				pos,
				edge_cmap = plt.cm.Blues,
				width=[g[u][v]['attr']['width'] for u, v in g.edges],
				edge_color=[g[u][v]['attr']['color'] for u, v in g.edges],
				style=[g[u][v]['attr']['style'] for u, v in g.edges],
				alpha=0.5
	)
	for p in pos:  # raise text positions
   		pos[p][0] += 0.05 if pos[p][0] < 0.5 else -0.05
	nx.draw_networkx_labels(g, pos)

	plt.show()
	return
	edge_x = []
	edge_y = []
	for edge in g.edges():
		if 'pos' in g.nodes[edge[0]]:
			print('test12345')
			x0, y0 = g.nodes[edge[0]]['pos']
			x1, y1 = g.nodes[edge[1]]['pos']
			edge_x.append(x0)
			edge_x.append(x1)
			edge_x.append(None)
			edge_y.append(y0)
			edge_y.append(y1)
			edge_y.append(None)

	edge_trace = go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=0.5, color='#888'),
		hoverinfo='none',
		mode='lines')

	node_x = []
	node_y = []
	for node in g.nodes():
		if 'pos' in g.nodes[node]:
			x, y = g.nodes[node]['pos']
			node_x.append(x)
			node_y.append(y)

	node_trace = go.Scatter(
		x=node_x, y=node_y,
		mode='markers',
		hoverinfo='text',
		marker=dict(
			showscale=True,
			# colorscale options
			#'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
			#'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
			#'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
			colorscale='YlGnBu',
			reversescale=True,
			color=[],
			size=10,
			colorbar=dict(
			    thickness=15,
			    title='Node Connections',
			    xanchor='left',
			    titleside='right'
			),
			line_width=2))

	node_adjacencies = []
	node_text = []
	for node, adjacencies in enumerate(g.adjacency()):
	    node_adjacencies.append(len(adjacencies[1]))
	    node_text.append('# of connections: '+str(len(adjacencies[1])))

	node_trace.marker.color = node_adjacencies
	node_trace.text = node_text

	fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
	print('test1')
	fig.show()
	print('test2')


if __name__ == '__main__':
	create_graph()
