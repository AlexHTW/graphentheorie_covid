# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from colour import Color

import pandas as pd
from tqdm import tqdm
import time

from CoronaNet import CoronaNet
from helpers import get_unique_vals
from dateutil.rrule import rrule, DAILY
from RKI_covid19 import RKI_covid19

THIN_EDGE = 0.3
THICK_EDGE = 3
MIN_R_VALUE = 0.0
MAX_R_VALUE = 3

N_COLORS = 50
COLORS = list(Color("blue").range_to(Color("red"), N_COLORS))
COLORSCALE = [((i / N_COLORS), color.get_hex())
              for i, color in enumerate(COLORS)]

TARGET_DATE = date(2021,1,15)#date.today() - timedelta(days=3)
FIRST_DAY = date(2020, 4, 1)

CoronaNet.FIRST_DAY = FIRST_DAY
RKI_covid19.FIRST_DAY =FIRST_DAY

CoronaNet.TARGET_DATE = TARGET_DATE
RKI_covid19.TARGET_DATE = TARGET_DATE


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
    

def get_size_for_number_of_cases(number_of_cases):
    """
            for given number_of_cases it returns size of node
    """
    i = 0
    if 0 <= number_of_cases < 10:
        i = 1
    elif 10 <= number_of_cases <25:
        i = 2
    elif 25 <= number_of_cases < 50:
        i = 3
    elif 50 <= number_of_cases < 100:
        i = 4
    elif 100 <= number_of_cases < 200:
        i = 5
    elif 200 <= number_of_cases < 300:
        i = 6
    elif 300 <= number_of_cases <400:
        i = 7
    elif 400 <= number_of_cases <500: 
        i = 8

    size = 10 + i*0.5*5
    return int(size)


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
    #max_cases = cases['AnzahlFall_7_tage_100k'].max()  # ToDo: not working, because for some days its 0?
    m_ticktext = np.linspace(0, 2, num=30)
    m_tickvals = [get_color_for_r_value(x) for x in m_ticktext]
    cndata = cn.load_data_for_day(day)
    cndata.dropna(thresh=2, inplace=True)
    started_within_last_2w = cndata[(cndata.timespan == 'started_within_last_2w')].drop_duplicates(subset=['target_province', 'type_sub_cat', 'type'], keep='first')
    ongoing_2w_4w = cndata[(cndata.timespan == 'ongoing_2w_4w')].drop_duplicates().drop_duplicates(subset=['target_province', 'type_sub_cat', 'type'], keep='first')
    ongoing_4w = cndata[(cndata.timespan == 'ongoing_4w')].drop_duplicates().drop_duplicates(subset=['target_province', 'type_sub_cat', 'type'], keep='first')
    ended_within_2w = cndata[(cndata.timespan == 'ended_within_2w')].drop_duplicates().drop_duplicates(subset=['target_province', 'type_sub_cat', 'type'], keep='first')
    ended_within_2w_4w = cndata[(cndata.timespan == 'ended_within_2w_4w')].drop_duplicates().drop_duplicates(subset=['target_province', 'type_sub_cat', 'type'], keep='first')

    if not type(day) is date:
        day = day.date()

    # NODES

    # BUNDESLÄNDER
    nodes_state = []
    for i, node in enumerate(cn.u_provinces):
        x = 0.25
        y = i * cn.normalized_unit_y_provinces if not node == 'Countrywide' else (
            i + 2) * cn.normalized_unit_y_provinces  # ToDo: Place Countrywide
        r_value = cases.loc[node]['R-Wert']
        num_of_infec = cases.loc[node]['AnzahlFall_7_tage_100k']
        hovertemplate = "{0}<br>".format(node)
        hovertemplate += "started within last 2 weeks: {0}<br>".format(len(started_within_last_2w[(started_within_last_2w.target_province == node)].index))
        hovertemplate += "ongoing between 2 and 4 weeks: {0}<br>".format(len(ongoing_2w_4w[(ongoing_2w_4w.target_province == node)].index))
        hovertemplate += "ongoing since more than 4 weeks: {0}<br>".format(len(ongoing_4w[(ongoing_4w.target_province == node)].index))
        hovertemplate += "ended within last 2 weeks: {0}<br>".format(len(ended_within_2w[(ended_within_2w.target_province == node)].index))
        hovertemplate += "ended within last 2 to 4 weeks: {0}<br>".format(len(ended_within_2w_4w[(ended_within_2w_4w.target_province == node)].index))

        nodes_state.append({
            'key': node,
            'type': 'province',
            'x': x,
            'y': y,
            'textpos': "middle left",
            'r_value': r_value,
            'color': get_color_for_r_value(r_value),
            'hovertext': hovertemplate + 'R-Value: {0}<br>Number of cases per 100k population: {1}'.format(r_value, num_of_infec),
            'size': get_size_for_number_of_cases(num_of_infec),
        })
    node_trace_state = go.Scatter(
        x=[node['x'] for node in nodes_state],
        y=[node['y'] for node in nodes_state],
        text=[node['key'] for node in nodes_state],  # Labels
        textposition=[node['textpos'] for node in nodes_state],
        mode='markers+text',
        hoverinfo='name+text',
        hovertext= [node['hovertext'] for node in nodes_state],
        name = 'state node (size - 7 days incidence)',
        #showlegend=,
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale=COLORSCALE,#'Portland',
            reversescale=True,
            autocolorscale=False,
            cmin=0.2,
            cmax=2.5,
            color=[node['r_value'] for node in nodes_state],
            size=[node['size'] for node in nodes_state],
            colorbar=dict(
                thickness=15,
                title='R Value',
                xanchor='left',
                titleside='right',
                # tickvals= m_tickvals,
                # ticktext= [round(x,1) for x in m_ticktext]
            ),
            line_width=2)
    )

    nodes_measures = []
    # CREATE NODE FOR EACH TYPES / KATEGORIEN

    for i_type, node in enumerate(cn.u_types):
        # CREATE NODE FOR EACH SUBTYPES / SUBKATEGORIEN
        row_data = cn.data[(cn.data.type.isin([node]))]

        # print('types...')
        # print(type(day))
        # for idx, r in row_data.iterrows():
        #     print(type(r['date_start']))
        #     print(type(r['date_end']))

        temp_subtypes = get_unique_vals(row_data, 'type_sub_cat')
        base_y = i_type * cn.normalized_unit_y_types  # base y = ypos of Typenode
        for i_subtype, subnode in enumerate(temp_subtypes):
            exists = ((cndata[(cndata.type.isin([node]) & cndata.type_sub_cat.isin(
                [subnode]))]).shape)[0]  # is there any row containing both values?
            if exists:
                # hovertext -> Submaßnahmen
                hovertemplate = ''.join(
                    "Introduced from {0} to {1} at {2} <br>".format(
                        r['date_start'],
                        r['date_end'] if not pd.isna(r['date_end']) else "not specified",
                        r['target_province'],
                    )
                    for idx, r in row_data.drop_duplicates().iterrows()
                    if (r['date_end'] >= day or pd.isna(r['date_end']))
                    and r['date_start'] <= day
                    and r['type_sub_cat'] == subnode
                )
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
                nodes_measures.append({
                    'key': subnode,
                    'subkey': node,
                    'type': 'subtype',
                    'x': x,
                    'y': y,
                    'textpos': "middle right",
                    'color': "#909090",
                    'hovertext': subnode + "<br><br>" + hovertemplate,
                    'size': 10 , # ToDo: Entscheiden welche größe subkategorien haben,
                    'name' : 'measures node'
                })

        exists = (
            cndata[(cndata.type.isin([node]))].shape)[0]
        x = 0.70
        y = base_y

        # hovertext -> Maßnahmen
        hovertemplate = ''.join(
            "Introduced from {0} to {1} at {2} <br>".format(
                r['date_start'],
                r['date_end'] if not pd.isna(r['date_end']) else "not specified",
                r['target_province'],
            )
            for idx, r in row_data.drop_duplicates().iterrows()
            if (r['date_end'] >= day or pd.isna(r['date_end']))
            and r['date_start'] <= day
        )

        nodes_measures.append({
            'key': node,
            'type': 'type',
            'x': x,
            'y': y,
            'textpos': "middle right",
            'color': "black" if exists else "#A0A0A0",
            'hovertext': node + "<br><br>" + hovertemplate,
            'size': 30,
        })

    # fix y positions for subtypes (optional)
    y_subtypes = 1 / len([node for node in nodes_measures if node['type'] == "subtype"])
    y_iter = 0
    for node in nodes_measures:
        if node['type'] == "subtype":
            node['y'] = y_iter * y_subtypes
            y_iter += 1

    node_trace_measures = go.Scatter(
        x=[node['x'] for node in nodes_measures],
        y=[node['y'] for node in nodes_measures],
        text=[node['key'] for node in nodes_measures],  # Labels
        textposition=[node['textpos'] for node in nodes_measures],
        mode='markers+text',
        hoverinfo='name+text',
        hovertext= [node['hovertext'] for node in nodes_measures],
        name = 'measures node',
        marker=dict(
            #showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale=COLORSCALE,
            reversescale=True,
            color=[node['color'] for node in nodes_measures],
            size=[node['size'] for node in nodes_measures],
            line_width=2,
            )
    )

    nodes = []
    nodes.extend(nodes_state)
    nodes.extend(nodes_measures)
    # EDGES

    edges = []

    def draw_edges(data, edges, width, color, type_name, dash='solid'):
        subcatcoords = []
        catcoords = []
        edge_x = []
        edge_y = []
        hovertexts = []
        for idx, row in data.iterrows():
            x0 = get_node_attr_by_key(
                nodes=nodes, key=row['target_province'], attr="x")
            y0 = get_node_attr_by_key(
                nodes=nodes, key=row['target_province'], attr="y")
            x1 = get_node_attr_by_key(
                nodes=nodes, key=row['type_sub_cat'], attr="x", subkey=row['type'])
            y1 = get_node_attr_by_key(
                nodes=nodes, key=row['type_sub_cat'], attr="y", subkey=row['type'])
            if not [x0, x1, y0, y1] in subcatcoords:
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
                subcatcoords.append([x0, x1, y0, y1])
            x0 = get_node_attr_by_key(
                nodes=nodes, key=row['type_sub_cat'], attr="x", subkey=row['type'])
            y0 = get_node_attr_by_key(
                nodes=nodes, key=row['type_sub_cat'], attr="y", subkey=row['type'])
            x1 = get_node_attr_by_key(
                nodes=nodes, key=row['type'], attr="x")
            y1 = get_node_attr_by_key(
                nodes=nodes, key=row['type'], attr="y")
            if not [x0, x1, y0, y1] in catcoords:
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
                catcoords.append([x0, x1, y0, y1])

        edges.append(go.Scatter(
            x=edge_x, y=edge_y,
            line=go.scatter.Line(width=width, color=color, dash=dash),
            hoverinfo='none',
            mode='lines',
            name = type_name)
        )

    # started within last 2 weeks
    draw_edges(started_within_last_2w, edges,
               width=0.5, color='#888',  type_name= 'started within last 2 weeks', dash='dash',)
    # ongoing 2 to 4 weeks
    draw_edges(ongoing_2w_4w, edges, width=0.5, color='#888', type_name= 'ongoing between 2 and 4 weeks')

    # ongoing 4 weeks or more
    draw_edges(ongoing_4w, edges, width=1, color='#888', type_name= 'ongoing since 4 weeks or more')

    # ended within 2 weeks
    draw_edges(ended_within_2w, edges, width=0.5, color='#e88574', type_name= 'ended within last 2 weeks')

    # ended within 2 to 4 weeks
    draw_edges(ended_within_2w_4w, edges, width=0.5,
               color='#e88574', type_name= 'ended within last 2 to 4 weeks', dash='dash')

    return [node_trace_state, node_trace_measures, *edges]

def create_graph():
    # generate current types

    cn = CoronaNet()
    cases_dataset = RKI_covid19()
    ##############
    ### PLOTLY ###
    ##############

    start = time.time()

    frames = [
        go.Frame(
            data=create_edges_and_nodes(cn=cn, cases_dataset=cases_dataset, day=day),
            name=str(day)
        )
        for day in tqdm(rrule(DAILY, dtstart=CoronaNet.FIRST_DAY, until=TARGET_DATE))
    ]

    fig = go.Figure(
        data=create_edges_and_nodes(cn=cn, cases_dataset=cases_dataset, day = cn.FIRST_DAY),
        layout=go.Layout(
            title='CoronaNet Visualization',
            titlefont_size=16,
            showlegend=True,
            legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="right",
                    x=1
                ),
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
                            "args": [None, {"frame": {"duration": 200, "redraw": False},
                                            "fromcurrent": True, "transition": {"duration": 100,
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
                        "prefix": "Selected Date:",
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
                        } for day in tqdm(rrule(DAILY, dtstart=CoronaNet.FIRST_DAY, until=TARGET_DATE))
                    ]
                }
            ]
        ),
        frames=frames
    )
    fig.show()


if __name__ == '__main__':
    create_graph()
