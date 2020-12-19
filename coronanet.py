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

import difflib


THIN_EDGE = 0.3
THICK_EDGE = 3
MIN_R_VALUE = 0.5
MAX_R_VALUE = 1.4

N_COLORS = 50
COLORS = list(Color("green").range_to(Color("red"), N_COLORS))
COLORSCALE = [((i / N_COLORS), color.get_hex())
              for i, color in enumerate(COLORS)]

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


def correct_data_with_days(data):
    n_data = data
    for ind, row in data.iterrows():
        if(ind < len(data)-2):
            if row['Bundesland'] == data['Bundesland'][ind+1]:
                while True:
                    if row['Meldedatum'] < data['Meldedatum'][ind+1] - timedelta(days=1):
                        row['Meldedatum'] = row['Meldedatum'] + \
                            timedelta(days=1)
                        row['AnzahlFall'] = 0
                        n_data = n_data.append(row).sort_index()
                    else:
                        break
    # print(n_data)
    return n_data.sort_values(['Bundesland', 'Meldedatum']).reset_index(drop=True)


def get_cases_7_days(x, data):
    con1 = data['Meldedatum'] <= x['Meldedatum']
    con2 = data['Meldedatum'] > x['Meldedatum'] - timedelta(days=7)
    con3 = data['Bundesland'] == x['Bundesland']
    return data[con1 & con2 & con3]['AnzahlFall'].sum()


def get_cases_7_days_100k(x, data):
    ewz = data.loc[x['Bundesland']]['LAN_ew_EWZ']
    res = (x["AnzahlFall_7_tage_absolut"] * 100000) / ewz
    return res


def get_cases_s_4(x, data):
    con3 = data['Meldedatum'] >= x['Meldedatum'] - timedelta(days=10)
    con4 = data['Meldedatum'] <= x['Meldedatum'] - timedelta(days=4)
    con5 = data['Bundesland'] == x['Bundesland']
    return data[con3 & con4 & con5]['AnzahlFall'].sum()


def get_r_value_intervall_7_days(x):
    s_t = x['AnzahlFall_7_tage_absolut']
    s_t_4 = x['AnzahlFall_s_4']
    if s_t_4 == 0:
        return 0
    else:
        return s_t / s_t_4


def get_cases_data_json():
    response = requests.get(
        'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0/query?where=1%3D1&outFields=Bundesland,AnzahlFall,Meldedatum&outSR=800000000&f=json')
    data_json = json.loads(response.text)['features']

    data = pd.json_normalize(data_json, sep='_')
    data['attributes_Meldedatum'] = pd.to_datetime(
        data['attributes_Meldedatum'], unit='ms')
    # print(len(data))
    data = data.groupby(['attributes_Bundesland', 'attributes_Meldedatum'], as_index=False)[
        'attributes_AnzahlFall'].sum()

    print(data.head())
    data = data.sort_values(['attributes_Bundesland', 'attributes_Meldedatum'])
    print(data)

    #print(get_unique_vals(data, 'attributes_Bundesland'))

    a = data[(data['attributes_Bundesland'] == 'Hamburg')]
    # plt.show()
    print(data.info())
    print(len(a))


def get_bundesland_pop():
    bundesland_pop_data = pd.read_csv("bundesland.csv",
                                      encoding='utf8',
                                      usecols=['LAN_ew_GEN', 'LAN_ew_EWZ'], index_col='LAN_ew_GEN')
    bundesland_pop_data.loc['Countrywide'] = bundesland_pop_data.sum(
        numeric_only=True, axis=0)
    return bundesland_pop_data


def get_cases_data_csv():
    url = "https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv"
    data = pd.read_csv(url,
                       encoding='utf8',
                       usecols=['Bundesland', 'AnzahlFall', 'Meldedatum'],
                       parse_dates=['Meldedatum'])

    bundesland_cases = data.groupby(['Bundesland', 'Meldedatum'], as_index=False)[
        'AnzahlFall'].sum()
    bundesland_cases = bundesland_cases.sort_values(
        ['Bundesland', 'Meldedatum'])
    bundesland_cases['AnzahlFall_7_tage_absolut'] = bundesland_cases.apply(
        lambda x: get_cases_7_days(x, bundesland_cases), axis=1)
    bundesland_cases['AnzahlFall_s_4'] = bundesland_cases.apply(
        lambda x: get_cases_s_4(x, bundesland_cases), axis=1)
    return bundesland_cases


def get_coronanet_data():
    """
            gets 'live' data from github
    """
    country = "Germany"
    url = "https://raw.githubusercontent.com/saudiwin/corona_tscs/master/data/CoronaNet/data_country/coronanet_release/coronanet_release_{0}.csv".format(country)
    #url = "../coronanet_release_Germany.csv"
    data = pd.read_csv(url, encoding='iso-8859-1') #,error_bad_lines=False)
    #data["province"] = data["province"].str.decode('iso-8859-1').str.encode('utf-8')

    return data


def get_unique_vals(data, col="type"):
    """
            returns unique values of a column
            call to (e.g.) data.type.unique()
    """
    return getattr(getattr(data, col), 'unique')().tolist()


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


def find_best_match(outlier, targetlist):
    """
            returns best match (= most similar word) for outlier from target list
    """
    hits = []
    for i, word in enumerate(targetlist):
        similarity = difflib.SequenceMatcher(
            None, outlier.lower(), word.lower()).ratio()
        hits.append(similarity)

    # Index of highest Value in hits
    idx_max = max(range(len(hits)), key=hits.__getitem__)

    return targetlist[idx_max]


def clean_bundeslaender(data):
    #data_all = data[(data.province.isin(['-', np.nan]))]
    data['target_province'] = data['target_province'].str.replace(';', '')
    data['target_province'] = data['target_province'].str.replace(
        r'^-$', 'Countrywide')
    data[(data.target_province.isin(['-', np.nan]))] = data[(
        data.target_province.isin(['-', np.nan]))].assign(target_province='Countrywide')

    # MANUAL CLEANING
    # first Step: add missing rows
    # ToDo: Speedup
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
    data = data[~(data.target_province.isin(["Berlin Brandenburg",
                                             "Bayern Baden-Württemberg", "Gütersloh, Warendorf", "Lombardy"]))]

    # iterate thru rows and set outliers of dictionary to most similar
    # ToDo: Speedup -> e.g. np.apply_along_axis(lambda x: find_best_match(x) if x not in R_VALUES.keys() else x, 1, data['target_province'])
    missmatches = []
    for idx, row in data.iterrows():
        if row['target_province'] not in list(R_VALUES.keys()):
            missmatches.append(row['target_province'])
            best_match = find_best_match(
                row['target_province'], list(R_VALUES.keys()))
            row['target_province'] = best_match
            data.append(row)
    data = data[~(data.target_province.isin(list(set(missmatches))))]

    return data


def clean_bundeslaender_2(data):
    data = data.replace(to_replace=r'^Baden-Württemberg',
                        value='Baden-Wuerttemberg', regex=True)
    data = data.replace(to_replace=r'^Thüringen',
                        value='Thuringia', regex=True)
    data = data.replace(to_replace=r'^Bayern', value='Bavaria', regex=True)
    data = data.replace(to_replace=r'^Niedersachsen',
                        value='Lower Saxony', regex=True)
    data = data.replace(to_replace=r'^Sachsen', value='Saxony', regex=True)
    data = data.replace(to_replace=r'^Nordrhein-Westfalen',
                        value='North Rhine-Westphalia', regex=True)
    data = data.replace(to_replace=r'^Hessen', value='Hesse', regex=True)
    return data

def save_calculated_number_cases_to_csv(filename,data, cal_date):
    if os.path.isfile(filename):
        saved_data_bc = pd.read_csv(filename).set_index('Meldedatum')
        if cal_date in saved_data_bc.index:
            saved_data_bc = saved_data_bc.drop(cal_date)
            saved_data_bc.to_csv(filename, header=True)
        else:
            data.to_csv(filename, mode='a', header=False)
    else: 
        data.to_csv(filename, header = True)

def create_graph():
    # all data
    data_measure = get_coronanet_data()  # provinces and measure
    data_measure = clean_bundeslaender(data_measure)
    bundesland_pop_data = get_bundesland_pop()  # provinces and population
    bundesland_pop_data.loc['Countrywide'] = bundesland_pop_data.sum(
        numeric_only=True, axis=0)
    cases = get_cases_data_csv().set_index('Meldedatum')  # provinces and cases

    data_measure[(data_measure.type_sub_cat.isin(['-', np.nan]))] = data_measure[(data_measure.type_sub_cat.isin(
        ['-', np.nan]))].assign(type_sub_cat='Not further spezified')  # ToDo: nan values for subcats

    # get provinces and measure types
    u_provinces = get_unique_vals(data_measure, 'target_province')
    u_provinces.sort()  # sort reverse alphabetically
    # put countrywide to end of list
    u_provinces.append(u_provinces.pop(u_provinces.index('Countrywide')))
    u_types = get_unique_vals(data_measure, 'type')
    u_subtypes = get_unique_vals(data_measure, 'type_sub_cat')

    # normalized unit y, so all provinces y-positions are evenly between 0 and 1
    normalized_unit_y_provinces = 1 / len(u_provinces)
    # normalized unit y, so all types y-positions are evenly between 0 and 1
    normalized_unit_y_types = 1 / len(u_types)
    #print("y_prov einheitslänge: {0}".format(normalized_unit_y_provinces))
    #print("y_types einheitslänge: {0}".format(normalized_unit_y_types))

    # print(u_provinces)

    data_measure['date_start'] = pd.to_datetime(
        data_measure['date_start']).dt.date
    data_measure['date_end'] = pd.to_datetime(data_measure['date_end']).dt.date

    # day of interest

    yesterday = date.today() - timedelta(days=2)
    selected_date = yesterday  # temporarily yesterday

    # generate current types

    date_2w_before = selected_date - timedelta(days=14)
    date_4w_before = selected_date - timedelta(days=28)
    date_2w_after_end = data_measure.date_end + timedelta(days=14)
    date_4w_after_end = data_measure.date_end + timedelta(days=28)

    started_within_last_2w = data_measure[data_measure.date_start > date_2w_before]
    ongoing_2w_4w = data_measure[(data_measure.date_start < date_2w_before) & (
        data_measure.date_start > date_4w_before) & (data_measure.date_end > selected_date)]
    ongoing_4w = data_measure[(data_measure.date_start < date_4w_before) & (
        data_measure.date_end > date_2w_before)]
    ended_within_2w = data_measure[(selected_date > data_measure.date_end) & (
        selected_date < date_2w_after_end)]
    ended_within_2w_4w = data_measure[(selected_date > date_2w_after_end) & (
        selected_date < date_4w_after_end)]

    currenttime_data = pd.concat([started_within_last_2w, ongoing_2w_4w, ongoing_4w, ended_within_2w, ended_within_2w_4w],
                                 ignore_index=True, sort=False)  # dataset including all the above datasets

    yesterday = date.today() - timedelta(days=2) 
    cal_date = '{0}-{1}-{2}'.format(yesterday.year,yesterday.month, yesterday.day)
    cases = cases.loc[cal_date]
    cases = cases.reset_index().set_index("Bundesland")
    cases.loc['Countrywide'] = cases.sum(numeric_only=True, axis=0)
    cases['Meldedatum'] = cases['Meldedatum'].dt.date
    cases.loc['Countrywide','Meldedatum'] = cal_date
    cases = cases.reset_index()
    cases['AnzahlFall_7_tage_100k'] = cases.apply(
        lambda x: get_cases_7_days_100k(x, bundesland_pop_data), axis=1)
    cases['R-Wert'] = cases.apply(
        lambda x: get_r_value_intervall_7_days(x), axis=1)
    cases = clean_bundeslaender_2(cases)
    cases = cases.reset_index(drop=True).set_index("Bundesland")
    max_cases = cases['AnzahlFall_7_tage_100k'].max()  # ToDo: -
    m_ticktext = np.linspace(cases['R-Wert'].min(), cases['R-Wert'].max(), num = 5)
    m_tickvals = [get_color_for_r_value(x) for x in m_ticktext]
    #print(cases)
    ##############
    ### PLOTLY ###
    ##############

    # NODES

    # BUNDESLÄNDER
    nodes = []
    for i, node in enumerate(u_provinces):
        x = 0.25
        y = i * normalized_unit_y_provinces if not node == 'Countrywide' else (
            i + 2) * normalized_unit_y_provinces  # ToDo: Place Countrywide
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
    for i_type, node in enumerate(u_types):
        # CREATE NODE FOR EACH SUBTYPES / SUBKATEGORIEN
        row_data = data_measure[(data_measure.type.isin([node]))]
        temp_subtypes = get_unique_vals(row_data, 'type_sub_cat')
        base_y = i_type * normalized_unit_y_types  # base y = ypos of Typenode
        for i_subtype, subnode in enumerate(temp_subtypes):
            exists = ((currenttime_data[(currenttime_data.type.isin([node]) & currenttime_data.type_sub_cat.isin(
                [subnode]))]).shape)[0]  # is there any row containing both values?
            if exists:
                #print(data[(data.type.isin([node]) & data.type_sub_cat.isin([subnode]))])
                x = 0.35
                # subtype length unit,
                # eg if theres 3 types and 2 subtypes:
                # normalized_unit_y_types is 0.33
                # normalized_unit_y_subtypes is 0.33 / 2 = 0.167
                normalized_unit_y_subtypes = normalized_unit_y_types / \
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
            currenttime_data[(currenttime_data.type.isin([node]))].shape)[0]
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
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="Data: <a href='https://www.coronanet-project.org/'> Coronanet Project</a> | <a href= 'https://npgeo-corona-npgeo-de.hub.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0'> RKI Covid-19 </a>",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    fig.show()


if __name__ == '__main__':
    create_graph()
