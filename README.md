# Graphentheorie_covid
## VC / HC AT Advanced Topics: Graphentheorie & Netzwerkanalyse WiSe2020/21 - HTW Berlin



### Das ist ein Repository für das Projekt "Exploration der Auswirkung von Corona-Maßnahmen europäischer Staaten auf die Ausbreitung des Corona-Virus"


## Ausführung

# optional: create venv
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# create folders to save data

mkdir data data_cases
 
# show graph
python main.py

# get dataset covid data
wget -O cases_2021-1-10.csv https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv

