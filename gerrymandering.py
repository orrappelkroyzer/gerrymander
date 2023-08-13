import os, sys
from pathlib import Path
local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)

from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json", output_dir_suffix="gerrymandering")
from datetime import datetime

from utils.plotly_utils import fix_and_write

import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from create_db_tools.processor_functions.static_and_demography import process_census
from create_db_tools.consts import southeast_states, southwest_states
from extended_utils.analyses.lr.basic import coeffs_analysis

def read_db():
    logger.info("Reading DB")
    results = {}
    for i in ['president']:
        logger.info(f"Reading {i}")
        by_cd_an_county_gdf = gpd.read_file(Path(config['db_dir']) / f'gerrymandering_{i}.geojson')
        by_cd_an_county_gdf['id'] = by_cd_an_county_gdf.index
        for field in ['wasted_votes_pct', 'excess_votes_pct', 'wasted_votes', 'excess_votes']:
            by_cd_an_county_gdf.loc[by_cd_an_county_gdf[field].notnull(), field] = by_cd_an_county_gdf.loc[by_cd_an_county_gdf[field].notnull(), field].apply(float)
        by_county = by_cd_an_county_gdf.groupby(['FIPS', 'year']).agg({'Votes' : 'sum', 
                                                                       'excess_votes' : 'sum', 
                                                                       'wasted_votes' : 'sum', 
                                                                       'R_votes' : 'sum',
                                                                       "D_votes" : 'sum'}).reset_index()
        by_county = by_county.rename(columns={'excess_votes' : 'Excess Votes', 'wasted_votes' : 'Wasted Votes'})
        logger.info(by_county.columns)
        for x in ['Excess Votes', 'Wasted Votes']:
            by_county[f'{x} %'] = by_county[x]/by_county['Votes']
            by_county[f'{x} (R)'] = by_county[x].apply(lambda x: max(x, 0))
            by_county[f'{x} (D)'] = by_county[x].apply(lambda x: max(-x, 0))
            by_county[f'{x} % (Both Parties)'] = by_county[f'{x} %'].apply(abs)
        
        
        
        by_county.loc[by_county['D_votes'] > by_county['R_votes'], 'general_EG'] = by_county['R_votes'] + by_county['Votes']/2 + by_county['D_votes']
        by_county['Efficiency Gap'] = ((by_county['Excess Votes (R)'] + by_county['Wasted Votes (R)']) - 
                           (by_county['Excess Votes (D)'] + by_county['Wasted Votes (D)'])) / by_county['Votes']
        by_county['Efficiency Gap_normalized'] = by_county['Efficiency Gap'] - (by_county['R_votes']-by_county['D_votes'])/ (2 * by_county['Votes'])
                          
        
        for x in ['Excess Votes %', 'Wasted Votes %', 'Efficiency Gap', 'Efficiency Gap_normalized']:                       
            by_county[f'{x} (Both Parties)'] = by_county[x].apply(abs)
            by_county[f'{x} (R)'] = by_county[x].apply(lambda x: max(x, 0))
            by_county[f'{x} (D)'] = by_county[x].apply(lambda x: max(-x, 0))
        by_county = add_info(by_county)
        by_county = add_interaction_fields(by_county)
        county_gdf = gpd.read_file(Path(config['db_dir']) / 'GIS' / "counties.shp").set_index('FIPS')
        results[i] = (by_cd_an_county_gdf, by_county, county_gdf)
    return results

def add_info(df):
    static_df = process_census()
    df = df.merge(static_df[['FIPS', 'Total Population', 'County']], on='FIPS', how='left')
    
    additional_data = pd.read_csv(Path(config['db_dir']) / "db_creation_input" / "Gerrymander_quan_data032723.csv")\
        .set_index('FIPS_full')
    additional_data['college_or_more'] = additional_data[['college', 'graduate']].sum(axis=1)
    additional_data['repcontrol_10'] = additional_data['repcontrol_10'].fillna(0)
    additional_data['repcontrol_20'] = additional_data['repcontrol_20'].fillna(0)
    additional_data = additional_data.rename(columns=ivs_dict)[ivs + ['State']]
    return df.join(additional_data, on='FIPS', how='left').drop_duplicates()

gerrymander_list = {
        'Partisan Gerrymander' : ['Florida', 'Gerogia', 'Indiana', 'Mischigan', 'North Carolina', 'Ohio', 
                                'Pennsylvania', 'South Carolina', 'Texas', 'Virginia', 'Maryland', 
                                'Massachussetts', 'California'],
        'Republican Gerrymander' : ['Florida', 'Gerogia', 'Indiana', 'Mischigan', 'North Carolina', 'Ohio', 
                                'Pennsylvania', 'South Carolina', 'Texas', 'Virginia'],
        'Democratic Gerrymander' : ['Maryland', 'Massachussetts', 'California'],
        'Majority Minority Districts' : {'African-American' : ['Alabama', 'Florida', 'Georgia', 'Illinois', 
                                                                'Louisiana', 'Maryland', 'Michigan', 'Mississippi', 
                                                                'New York', 'North Carolina', 'Ohio', 'Pensylvania', 'South Carolina', 
                                                                'Tennesee', 'Virginia'],
                                        'Hispanic' : ['Arizona', 'California', 'Florida', 'Illinois', 'New Jersey', 'New York', 'Texas']}}

def add_interaction_fields(df):
    
    for k, v in gerrymander_list.items():
        if type(v) == dict:
            for k2, v2 in v.items():
                df[f"{k2} {k}"] = df['State'].isin(v2)
                df[f"{k} * % {k2}"] = df[f"{k2} {k}"] * df[f"% {k2}"]
        else:
            df[k] = df['State'].isin(v)
            for k2 in ['African-American', 'Hispanic']:
                df[f"{k} * % {k2}"] = df[k] * df[f"% {k2}"]
    return df
                
# def by_cd_and_county_map(df, year, name):
#     t = pd.concat([df[['CD', 'County', 'State', 'id', 'wasted_votes_pct']].rename(columns={'wasted_votes_pct' : 'val'}).assign(var='wasted_votes_pct'),
#         df[['CD', 'County', 'State', 'id', 'excess_votes_pct']].rename(columns={'excess_votes_pct' : 'val'}).assign(var='excess_votes_pct')])
#     t = t[t['val'].notnull()]
#     t['val'] = t['val'].apply(float)
#     fig = px.choropleth(t,
#                     geojson=gpd.GeoDataFrame(df[['geometry']]),
#                     locations='id',
#                     color='val',
#                     facet_row='var',
#                     projection='albers usa',
#                     color_continuous_scale='RdBu_r',
#                     hover_data=['val', 'County', 'State', 'CD'],
#                     title=f"{name.title()} {year}",
#                     color_continuous_midpoint=0)
#     fix_and_write(fig=fig,
#                  filename=f"by_county_and_cd_{name}_{year}",
#                  output_dir=config['output_dir'] / "maps",
#                  output_type='html')


# def by_county_map(df, gdf, year, name):
#     t = pd.concat([df[['FIPS', 'County', 'State', 'wasted_votes_pct']].rename(columns={'wasted_votes_pct' : 'val'}).assign(var='wasted_votes_pct'),
#         df[['FIPS', 'County', 'State', 'excess_votes_pct']].rename(columns={'excess_votes_pct' : 'val'}).assign(var='excess_votes_pct')])
#     t = t[t['val'].notnull()]
#     t['val'] = t['val'].apply(float)
#     fig = px.choropleth(t,
#                     geojson=gdf,
#                     locations='FIPS',
#                     color='val',
#                     facet_row='var',
#                     projection='albers usa',
#                     color_continuous_scale='RdBu_r',
#                     hover_data=['val', 'County', 'State'],
#                     title=f"{name.title()} {year}",
#                     color_continuous_midpoint=0)
#     fix_and_write(fig=fig,
#                  filename=f"by_county_{name}_{year}",
#                  output_dir=config['output_dir'] / "maps",
#                  output_type='html')
#     for val in ['EG', 'EG_normalized']:
#         t = df[df[val].notnull()]
#         t[val] = t[val].apply(float)
#         fig = px.choropleth(t,
#                         geojson=gdf,
#                         locations='FIPS',
#                         color=val,
#                         projection='albers usa',
#                         color_continuous_scale='RdBu_r',
#                         hover_data=[val, 'County', 'State'],
#                         title=f"Efficiency Gap {name.title()} {year}",
#                         color_continuous_midpoint=0)
#         fix_and_write(fig=fig,
#                     filename=f"{val}_{name}_{year}",
#                     output_dir=config['output_dir'] / "maps",
#                     output_type='html')


ivs_dict = {
    'repcontrol_10' : 'GOP Control 2010',
    'repcontrol_20' : 'GOP Control 2020',
    'college_or_more' : '% Wth College Degree',
    'Black' : '% African-American',
    'Hispanic': '% Hispanic',
    'sec5' : 'Section 5',
    'southeast' : 'Southeastern States',
    'southwest' : 'Southwestern States',
    #'trump20' : '% Trump 2020'
    }




ivs = list(ivs_dict.values())
ivs.sort()

# def maps(dfs):
#     for name, (by_cd_and_county, by_county, gdf) in dfs.items():
#             for year in by_cd_and_county['year'].unique():
#                 by_cd_and_county_map(by_cd_and_county[by_cd_and_county.year == year], year, name)
#                 by_county_map(by_county[by_county.year == year], gdf, year, name)

# def regression(dfs):
#     for k in ['Majority Minority Districts']: # ["base"] + list(gerrymander_list.keys()):
#         if k == 'base':
#             t_ivs = ivs
#         elif k == 'Majority Minority Districts':
#             t_ivs = ivs + ['African-American Majority Minority Districts', 
#                            'Hispanic Majority Minority Districts',
#                            'Majority Minority Districts * % African-American',
#                            'Majority Minority Districts * % Hispanic']
#         else:
#             t_ivs = ivs + [k, f"{k} * % African-American", f"{k} * % Hispanic"]
#         output_dir = config['output_dir'] / k
#         for name, (by_cd_and_county, by_county, gdf) in dfs.items():
#             for year in by_cd_and_county['year'].unique():
#                 for dv in [['wasted_votes_pct', 'excess_votes_pct'], ['EG'], ['EG_normalized']]:
#                     if year in [2016, 2020]:
#                         coeffs_analysis(df=by_county[by_county.year == year],
#                                         output_dir=output_dir / "abs",
#                                         dependent_variables=[f'{x}_abs' for x in dv],
#                                         independent_variables=t_ivs,
#                                         filename=f"{name}_{year}_{'+'.join(dv)}",
#                                         sort_values=False,
#                                         weights_variable='Total Population')
#                         coeffs_analysis(df=by_county[by_county.year == year],
#                                         output_dir=output_dir / "R",
#                                         dependent_variables=[f'{x}_R' for x in dv],
#                                         independent_variables=t_ivs,
#                                         filename=f"{name}_{year}_{'+'.join(dv)}",
#                                         sort_values=False,
#                                         weights_variable='Total Population')
#                         coeffs_analysis(df=by_county[by_county.year == year],
#                                         output_dir=output_dir / "D",
#                                         dependent_variables=[f'{x}_D' for x in dv],
#                                         independent_variables=t_ivs,
#                                         filename=f"{name}_{year}_{'+'.join(dv)}",
#                                         sort_values=False,
#                                         weights_variable='Total Population')

#                     coeffs_analysis(df=by_county[by_county.year.isin([2016, 2020])],
#                                         output_dir=output_dir / "abs",
#                                         dependent_variables=[f'{x}_abs' for x in dv],
#                                         independent_variables=t_ivs,
#                                         sort_values=False,
#                                         filename=f"{name}_presidential_years_{'+'.join(dv)}",
#                                         weights_variable='Total Population')
#                     coeffs_analysis(df=by_county[by_county.year.isin([2016, 2020])],
#                                         output_dir=output_dir / "R",
#                                         dependent_variables=[f'{x}_R' for x in dv],
#                                         independent_variables=t_ivs,
#                                         sort_values=False,
#                                         filename=f"{name}_presidential_years_{'+'.join(dv)}",
#                                         weights_variable='Total Population')
#                     coeffs_analysis(df=by_county[by_county.year.isin([2016, 2020])],
#                                         output_dir=output_dir / "D",
#                                         dependent_variables=[f'{x}_D' for x in dv],
#                                         independent_variables=t_ivs,
#                                         sort_values=False,
#                                         filename=f"{name}_presidential_years_coefficients",
#                                         weights_variable='Total Population')

def maps(dfs):
    name = 'president'
    by_cd_and_county, by_county, gdf = dfs[name]
    df = by_county[by_county.year == 2020]
    df = df[df['Efficiency Gap'].notnull()]
    df['Efficiency Gap'] = df['Efficiency Gap'].apply(float)
    df['EG_raw'] = (df['Efficiency Gap'] * df['Votes']).apply(abs)
    df = df.join(gdf['geometry'], on='FIPS')
    df = df.join(df['geometry'].apply(lambda x: pd.Series(x.centroid.coords[0], index=['long', 'lat']))).drop(columns='geometry')
    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=df['long'],
            lat=df['lat'],
    #        text=aggregated_data['county_fips'],
            marker=dict(
                size=df['EG_raw'],
                line=dict(width=0.5, color="black"),
                sizemode='area',
                sizeref=2.0 * max(df['EG_raw']) / (66.0 ** 2),  # Adjust the size scaling
                color=df['Efficiency Gap'],
                colorscale='RdBu_r',  # Choose any colorscale you like
                colorbar=dict(title="Efficiency Gap"),
            ),
        )
    )
    fig.update_geos(scope='usa')
    fix_and_write(fig=fig,
                    filename=f"US",
                    output_dir=config['output_dir'])
    t = df[df['State'].isin(['California', 'Nevada'])]
    fig = px.choropleth(t,
                        geojson=gdf,
                        locations='FIPS',
                        color='Efficiency Gap',
                        projection='albers usa',
                        color_continuous_scale='RdBu_r',
                        hover_data=['Efficiency Gap', 'County', 'State'],
                        title=f"Efficiency Gap (2020)",
                        color_continuous_midpoint=0)
    center_lat = 36.7783
    center_lon = -119.4179
    zoom_level = 2

    fig.update_geos(
        center=dict(lon=center_lon, lat=center_lat),
        projection_scale=zoom_level,
    )
    fix_and_write(fig=fig,
                    filename=f"CA_NV",
                    output_dir=config['output_dir'])

def regression(dfs):
    name = 'president'
    by_cd_and_county, by_county, gdf = dfs[name]
    t_ivs = ivs + ['African-American Majority Minority Districts', 
                           'Hispanic Majority Minority Districts',
                           'Majority Minority Districts * % African-American',
                           'Majority Minority Districts * % Hispanic']
    df = by_county[by_county.year.isin([2016, 2020])]
    coeffs_analysis(df=df,
                    output_dir=config['output_dir'],
                    dependent_variables=['Efficiency Gap (Both Parties)'],
                    independent_variables=t_ivs,
                    sort_values=False,
                    filename='EG_abs',
                    weights_variable='Total Population')
    coeffs_analysis(df=df,
                    output_dir=config['output_dir'],
                    dependent_variables=['Efficiency Gap (R)', 'Efficiency Gap (D)'],
                    independent_variables=t_ivs,
                    sort_values=False,
                    filename='EG_party',
                    weights_variable='Total Population')
    coeffs_analysis(df=df,
                    output_dir=config['output_dir'],
                    dependent_variables=['Excess Votes (R)', 'Excess Votes (D)',
                                         'Wasted Votes (R)', 'Wasted Votes (D)'],
                    independent_variables=t_ivs,
                    sort_values=False,
                    filename='wasted_excess',
                    height_factor=2,
                    weights_variable='Total Population')



def main():
    dfs = read_db()
    #maps(dfs)
    regression(dfs)



if __name__ == "__main__":
    main()
