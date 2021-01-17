# -*- coding: utf-8 -*-
from datetime import datetime, date, timedelta
import json
import random
import os
from dateutil.rrule import rrule, DAILY

import xarray
import pandas as pd
import numpy as np
from colour import Color
from pandas import DataFrame
import requests
import difflib

from helpers import get_unique_vals, find_best_match

class CoronaNet:
    FIRST_DAY = date(2020,4,1)
    TARGET_DATE = date.today() - timedelta(days=2)
    DATADIRPATH = os.path.join(os.path.dirname(__file__), 'data')
    BUNDESLAENDERNODES = [
        'Baden-Wuerttemberg', 'Bremen', 'Mecklenburg-Vorpommern', 'Bavaria',
        'Lower Saxony', 'North Rhine-Westphalia', 'Saxony-Anhalt', 'Hamburg',
        'Schleswig-Holstein', 'Hesse', 'Rheinland-Pfalz', 'Saarland', 'Saxony',
        'Thuringia', 'Berlin','Brandenburg', 'Lombardy', 'Countrywide'
    ]
    COLUMNS = ['date_start', 'date_end', 'type', 'type_sub_cat', 'target_province']
    DATASETDESCRIPTORS = [
        "started_within_last_2w", "ongoing_2w_4w", "ongoing_4w",
        "ended_within_2w","ended_within_2w_4w"
    ]

    def __init__(self, update = False):
        self.get_coronanet_dataset()
        if update:
            self.update_offlinedata()

    def get_full_container(self, update = False):
        data = None
        for day in rrule(DAILY, dtstart = CoronaNet.FIRST_DAY, until=CoronaNet.TARGET_DATE):
            if data is not None:
                temp = self.load_data_for_day(day.date(), update = update)
                temp.loc[:, 'day'] = pd.Series([str(day.date())] * temp.shape[0], index=temp.index)
                data = pd.concat([
                    data, temp
                ], ignore_index=True)
            else:
                data = self.load_data_for_day(day.date(), update = update)
                data.loc[:, 'day'] = pd.Series([str(day.date())] * data.shape[0], index=data.index)
        self.data['day'] = pd.to_datetime(self.data['day']).dt.date
        return data

    def load_data_for_day(self, day, update=False):
        filepath = os.path.join(CoronaNet.DATADIRPATH, "{0}_{1}_{2}.csv".format(day.year, day.month, day.day))
        if os.path.isfile(filepath) and not update:
            datacontainer = pd.read_csv(filepath)
        else:
            datacontainer = self.generate_data_for_day(day)
            datacontainer.to_csv(filepath, index=False)

        return datacontainer


    def update_offlinedata(self):
        for day in rrule(DAILY, dtstart = CoronaNet.FIRST_DAY, until=date.today() - timedelta(days=2)):
            self.load_data_for_day(day.date(), update = True)

    def generate_data_for_day(self, day):
        # set needed variables
        self.get_coronanet_dataset()

        # return self.data

        ##########################
        ### CREATE SUBDATASETS ###
        ##########################

        # clean datetime columns
        self.data['date_start'] = pd.to_datetime(self.data['date_start']).dt.date
        self.data['date_end'] = pd.to_datetime(self.data['date_end']).dt.date

        # timespans
        day = day.date()
        date_2w_before = day - timedelta(days=14)
        date_4w_before = day - timedelta(days=28)
        date_2w_after_end = day + timedelta(days=14)
        date_4w_after_end = day + timedelta(days=28)

        # create subdatasets for timespans
        started_within_last_2w = self.data[self.data.date_start > date_2w_before]
        ongoing_2w_4w = self.data[(self.data.date_start < date_2w_before) & (
                self.data.date_start > date_4w_before) & (self.data.date_end > day)]
        ongoing_4w = self.data[(self.data.date_start < date_4w_before) & (
                self.data.date_end > date_2w_before)]
        ended_within_2w = self.data[(self.data.date_end < day) & (
                self.data.date_end > date_2w_before)]
        ended_within_2w_4w = self.data[(self.data.date_end < date_2w_before) & (
                self.data.date_end > date_4w_before)]

        started_within_last_2w['timespan'] = 'started_within_last_2w'
        ongoing_2w_4w['timespan'] = 'ongoing_2w_4w'
        ongoing_4w['timespan'] = 'ongoing_4w'
        ended_within_2w['timespan'] = 'ended_within_2w'
        ended_within_2w_4w['timespan'] = 'ended_within_2w_4w'

        # concat
        datacontainer = pd.concat([
            started_within_last_2w, ongoing_2w_4w, ongoing_4w, ended_within_2w, ended_within_2w_4w
        ], ignore_index=True)

        return datacontainer

    def get_coronanet_dataset(self):
        """
                    gets 'live' data from github
                    and sets important instance variables
            """
        if not hasattr(self, 'data') \
            or not hasattr(self, 'u_provinces') \
            or not hasattr(self, 'u_types') \
            or not hasattr(self, 'u_subtypes') \
            or not hasattr(self, 'normalized_unit_y_provinces') \
            or not hasattr(self, 'normalized_unit_y_types'):
            country = "Germany"
            url = "https://raw.githubusercontent.com/saudiwin/corona_tscs/master/data/CoronaNet/data_country/coronanet_release/coronanet_release_{0}.csv".format(
                country)
            data = pd.read_csv(url, encoding='iso-8859-1')  # ,error_bad_lines=False)
            # data["province"] = data["province"].str.decode('iso-8859-1').str.encode('utf-8')

            print('data: ')
            print(data)
            print('describe target_province')
            print(data['target_province'].describe(include=[object]))
            print('describe ')
            print(data['target_province'].describe(include=[object]))
            print('unique bundesländer:')
            print(get_unique_vals(data, 'target_province'))
            print(len(get_unique_vals(data, 'target_province')))
            print('unique types:')
            print(get_unique_vals(data, 'type'))
            print(len(get_unique_vals(data, 'type')))
            print('unique type_sub_cats:')
            print(get_unique_vals(data, 'type_sub_cat'))
            print(len(get_unique_vals(data, 'type_sub_cat')))

            data = self.clean_bundeslaender(data)

            data[(data.type_sub_cat.isin(['-', np.nan]))] = data[(data.type_sub_cat.isin(
                ['-', np.nan]))].assign(type_sub_cat='Not further spezified')  # ToDo: nan values for subcats

            # set provinces and measure types
            u_provinces = get_unique_vals(data, 'target_province')
            u_provinces.sort()  # sort reverse alphabetically
            # put countrywide to end of list
            u_provinces.append(u_provinces.pop(u_provinces.index('Countrywide')))
            u_types = get_unique_vals(data, 'type')
            u_subtypes = get_unique_vals(data, 'type_sub_cat')

            # clean datetime columns
            data['date_start'] = pd.to_datetime(data['date_start']).dt.date
            data['date_end'] = pd.to_datetime(data['date_end']).dt.date

            data.drop(data.columns.difference(CoronaNet.COLUMNS), 1, inplace=True)

            self.data = data
            self.u_provinces = u_provinces
            self.u_types = u_types
            self.u_subtypes =u_subtypes

            # normalized unit y, so all provinces y-positions are evenly between 0 and 1
            self.normalized_unit_y_provinces = 1 / len(u_provinces)
            # normalized unit y, so all types y-positions are evenly between 0 and 1
            self.normalized_unit_y_types = 1 / len(u_types)

    def clean_bundeslaender(self, data):
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
            if row['target_province'] not in CoronaNet.BUNDESLAENDERNODES:
                missmatches.append(row['target_province'])
                best_match = find_best_match(row['target_province'], CoronaNet.BUNDESLAENDERNODES)
                row['target_province'] = best_match
                data.append(row)
        data = data[~(data.target_province.isin(list(set(missmatches))))]

        return data

if __name__ == "__main__":
    c = CoronaNet(update=True)