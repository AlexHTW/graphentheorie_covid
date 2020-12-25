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
from tqdm import tqdm

import difflib

from CoronaNet import CoronaNet
from helpers import get_unique_vals
from dateutil.rrule import rrule, DAILY
from RKI_covid19 import RKI_covid19

THIN_EDGE = 0.3
THICK_EDGE = 3
MIN_R_VALUE = 0.5
MAX_R_VALUE = 1.4

N_COLORS = 50
COLORS = list(Color("green").range_to(Color("red"), N_COLORS))
COLORSCALE = [((i / N_COLORS), color.get_hex())
              for i, color in enumerate(COLORS)]


def get_color_for_r_value(r_value):
    """
            for given r_value it returns corresponding color
    """
    old_min = MIN_R_VALUE
    old_max = MAX_R_VALUE
    new_min = 0
    new_max = N_COLORS - 1
    idx = (((r_value - old_min) * (new_max - new_min)) /
           (old_max - old_min)) + new_min
    return int(round(idx))
    

def get_size_for_number_of_cases(number_of_cases, max_cases):
    """
            for given number_of_cases it returns size of node
    """
    return 30*(number_of_cases / max_cases)


def get_node_attr_by_key(nodes, key, attr, subkey=None):
    """
            returns given attribute of element in nodes (=filtered by key)
    """
    if not subkey:
        possible_nodes = [node for node in nodes if node['key'] == key]
    else:
        possible_nodes = [node for node in nodes if node['key']
                          == key and node['subkey'] == subkey]
    if len(possible_nodes):
        return possible_nodes[0][attr]
    else:
        return None


def create_edges_and_nodes(cn, cases_dataset, day):
    cal_date = '{0}-{1}-{2}'.format(day.year, day.month, day.day)
    cases = cases_dataset.load_data_for_day(day).set_index('Bundesland')
    max_cases = cases['AnzahlFall_7_tage_100k'].max()  # ToDo: not working, because for some days its 0?
    m_ticktext = np.linspace(cases['R-Wert'].min(), cases['R-Wert'].max(), num=5)
    m_tickvals = [get_color_for_r_value(x) for x in m_ticktext]
    cndata = cn.load_data_for_day(day)
    cndata.dropna(thresh=2, inplace=True)
    started_within_last_2w = cndata[(cndata.timespan == 'started_within_last_2w')]
    ongoing_2w_4w = cndata[(cndata.timespan == 'ongoing_2w_4w')]
    ongoing_4w = cndata[(cndata.timespan == 'ongoing_4w')]
    ended_within_2w = cndata[(cndata.timespan == 'ended_within_2w')]
    ended_within_2w_4w = cndata[(cndata.timespan == 'ended_within_2w_4w')]

    # NODES

    # BUNDESLÄNDER
    nodes = []
    for i, node in enumerate(cn.u_provinces):
        x = 0.25
        y = i * cn.normalized_unit_y_provinces if not node == 'Countrywide' else (
            i + 2) * cn.normalized_unit_y_provinces  # ToDo: Place Countrywide
        # ToDo: FIX Exceptions -- KeyError: 'Hamburg'
        r_value = cases.loc[node]['R-Wert']
        num_of_infec = cases.loc[node]['AnzahlFall_7_tage_100k']
        nodes.append({
            'key': node,
            'type': 'province',
            'x': x,
            'y': y,
            'textpos': "middle left",
            'r_value': r_value,
            'color': get_color_for_r_value(r_value),
            'hovertext': 'R-Value: {0} , Number of cases: {1}'.format(r_value, num_of_infec),
            'size': get_size_for_number_of_cases(num_of_infec, max_cases)
        })

    # CREATE NODE FOR EACH TYPES / KATEGORIEN
    for i_type, node in enumerate(cn.u_types):
        # CREATE NODE FOR EACH SUBTYPES / SUBKATEGORIEN
        row_data = cn.data[(cn.data.type.isin([node]))]
        temp_subtypes = get_unique_vals(row_data, 'type_sub_cat')
        base_y = i_type * cn.normalized_unit_y_types  # base y = ypos of Typenode
        for i_subtype, subnode in enumerate(temp_subtypes):
            exists = ((cndata[(cndata.type.isin([node]) & cndata.type_sub_cat.isin(
                [subnode]))]).shape)[0]  # is there any row containing both values?
            if exists:
                #print(data[(data.type.isin([node]) & data.type_sub_cat.isin([subnode]))])
                x = 0.35
                # subtype length unit,
                # eg if theres 3 types and 2 subtypes:
                # normalized_unit_y_types is 0.33
                # normalized_unit_y_subtypes is 0.33 / 2 = 0.167
                normalized_unit_y_subtypes = cn.normalized_unit_y_types / \
                    len(temp_subtypes)
                #  base_y + sub length unit
                y = base_y + (i_subtype * normalized_unit_y_subtypes)
                nodes.append({
                    'key': subnode,
                    'subkey': node,
                    'type': 'subtype',
                    'x': x,
                    'y': y,
                    'textpos': "middle right",
                    'color': "#909090",
                    'hovertext': 'ToDo: Hovertext',  # ToDo: Hovertext Maßnahmensubkategorien
                    'size': 15  # ToDo: Entscheiden welche größe subkategorien haben
                })

        exists = (
            cndata[(cndata.type.isin([node]))].shape)[0]
        x = 0.70
        y = base_y
        nodes.append({
            'key': node,
            'type': 'type',
            'x': x,
            'y': y,
            'textpos': "middle right",
            'color': "black" if exists else "#A0A0A0",
            'hovertext': 'ToDo: Hovertext',  # ToDo: Hovertext Maßnahmenkategorien
            'size': 30
        })

    # fix y positions for subtypes (optional)
    y_subtypes = 1 / len([node for node in nodes if node['type'] == "subtype"])
    y_iter = 0
    for node in nodes:
        if node['type'] == "subtype":
            node['y'] = y_iter * y_subtypes
            y_iter += 1

    node_trace = go.Scatter(
        x=[node['x'] for node in nodes],
        y=[node['y'] for node in nodes],
        text=[node['key'] for node in nodes],  # Labels
        textposition=[node['textpos'] for node in nodes],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale=COLORSCALE,
            reversescale=True,
            color=[node['color'] for node in nodes],
            size=[node['size'] for node in nodes],
            colorbar=dict(
                thickness=15,
                title='R Value',
                xanchor='left',
                titleside='right',
                tickvals= m_tickvals,
                ticktext=  [round(x,3) for x in m_ticktext]
            ),
            line_width=2)
    )

    # EDGES

    edges = []

    def draw_edges(data, edges, width, color, dash='solid'):
        edge_x = []
        edge_y = []
        for idx, row in data.iterrows():
            x0 = get_node_attr_by_key(
                nodes=nodes, key=row['target_province'], attr="x")
            y0 = get_node_attr_by_key(
                nodes=nodes, key=row['target_province'], attr="y")
            x1 = get_node_attr_by_key(
                nodes=nodes, key=row['type_sub_cat'], attr="x", subkey=row['type'])
            y1 = get_node_attr_by_key(
                nodes=nodes, key=row['type_sub_cat'], attr="y", subkey=row['type'])
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            x0 = get_node_attr_by_key(
                nodes=nodes, key=row['type_sub_cat'], attr="x", subkey=row['type'])
            y0 = get_node_attr_by_key(
                nodes=nodes, key=row['type_sub_cat'], attr="y", subkey=row['type'])
            x1 = get_node_attr_by_key(
                nodes=nodes, key=row['type'], attr="x")
            y1 = get_node_attr_by_key(
                nodes=nodes, key=row['type'], attr="y")
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edges.append(go.Scatter(
            x=edge_x, y=edge_y,
            line=go.scatter.Line(width=width, color=color, dash=dash),
            hoverinfo='none',
            mode='lines')
        )

    # started within last 2 weeks
    draw_edges(started_within_last_2w, edges,
               width=0.5, color='#888', dash='dash')
    # ongoing 2 to 4 weeks
    draw_edges(ongoing_2w_4w, edges, width=0.5, color='#888')

    # ongoing 4 weeks or more
    draw_edges(ongoing_4w, edges, width=1, color='#888')

    # ended within 2 weeks
    draw_edges(ended_within_2w, edges, width=0.5, color='#e88574')

    # ended within 2 to 4 weeks
    draw_edges(ended_within_2w_4w, edges, width=0.5,
               color='#e88574', dash='dash')

    return [*edges, node_trace]

def create_graph():
    # generate current types

    cn = CoronaNet()
    cases_dataset = RKI_covid19()
    ##############
    ### PLOTLY ###
    ##############

    frames = [
        go.Frame(
            data=create_edges_and_nodes(cn=cn, cases_dataset=cases_dataset, day=day),
            name=str(day)
        )
        for day in tqdm(rrule(DAILY, dtstart=CoronaNet.FIRST_DAY, until=date.today() - timedelta(days=2)))
    ]

    fig = go.Figure(
        data=create_edges_and_nodes(cn=cn, cases_dataset=cases_dataset, day = cn.FIRST_DAY),
        layout=go.Layout(
            title='CoronaNet Visualization',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Data: <a href='https://www.coronanet-project.org/'> Coronanet Project</a> | <a href= 'https://npgeo-corona-npgeo-de.hub.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0'> RKI Covid-19 </a>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002)],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 500, "redraw": False},
                                            "fromcurrent": True, "transition": {"duration": 300,
                                                                                "easing": "quadratic-in-out"}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }
            ],
            sliders = [
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Year:",
                        "visible": True,
                        "xanchor": "right"
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args":
                                [
                                    [str(day)],
                                    {"frame": {"duration": 300, "redraw": False},
                                     "mode": "immediate",
                                     "transition": {"duration": 300}}
                                ],
                            "label": day.strftime('%d/%m/%Y'), # ToDo: label for current day...
                            "method": "animate"
                        } for day in tqdm(rrule(DAILY, dtstart=CoronaNet.FIRST_DAY, until=cn.TARGET_DATE))
                    ]
                }
            ]
        ),
        frames=frames
    )
    fig.show()


if __name__ == '__main__':
    create_graph()
