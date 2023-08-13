import os, sys
from pathlib import Path
local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-2])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)

from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json")
from datetime import datetime
from shapely.geometry import MultiPolygon



from create_db_tools.processor_functions.elections import process_2020_house_elections, process_2020_elections, \
                                                          process_2018_elections, \
                                                            process_2016_house_elections, process_2016_elections, \
                                                            process_2022_elections
                                                            

from create_db_tools.consts import urls, states_abbr, southeast_states
from create_db_tools.processor_functions.static_and_demography import process_fips_to_name
from shapely.errors import TopologicalError
from shapely.geometry import Polygon

import pandas as pd
import numpy as np
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
# warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)


def read_data(year, include_uncontested=True):
    if year == 2022:
        house_by_cd_and_county = process_2022_elections(include_uncontested)
        pres_by_cd_and_county = None

    elif year == 2020:
        house_by_cd_and_county = process_2020_house_elections(include_uncontested)
        pres_by_county, pres_by_cd, pres_cd_share_in_county, pres_by_cd_and_county, pres_presidential_votes_with_cd = process_2020_elections(with_cd=True)
        pres_by_cd_and_county = pres_by_cd_and_county.rename(columns={'2020 Votes' : 'Votes',
                                                                       '2020 Margin (%, Trump over Biden)' : 'R-D'})
        pres_by_cd_and_county.loc[:, 'votes_margin'] = pres_by_cd_and_county.loc[:, 'R-D'] * pres_by_cd_and_county.loc[:, 'Votes']
        
    elif year == 2018:
        house_by_cd_and_county = process_2018_elections(include_uncontested)
        pres_by_cd_and_county = None
    elif year == 2016:
        house_by_cd_and_county = process_2016_house_elections(include_uncontested)
        pres_by_county, pres_by_cd, pres_by_cd_and_county = process_2016_elections(with_cd=True)
        pres_by_cd_and_county.loc[:, 'votes_margin'] = pres_by_cd_and_county.loc[:, 'R-D'] * pres_by_cd_and_county.loc[:, 'Votes']
    
    house_by_cd_and_county = house_by_cd_and_county[['CD', 'FIPS', 'R-D', 'Votes']]
    house_by_cd_and_county.loc[:, 'votes_margin'] = house_by_cd_and_county.loc[:, 'R-D'] * house_by_cd_and_county.loc[:, 'Votes']
    
    return house_by_cd_and_county, pres_by_cd_and_county



def add_geometry(df):
    logger.info("Reading geojsons")
    gdf1 = gpd.read_file(Path(config['db_dir']) / "GIS" / "counties.shp")
    gdf1.index = gdf1.FIPS
    gdfs = {}
    logger.info(f"Reading 2016")
    gdfs[2016] = gpd.read_file(Path(config['db_dir']) / "GIS" / "tl_2016_us_cd115.shp").to_crs("WGS 84")
    fips2name = process_fips_to_name()
    fips2state = fips2name.groupby('State')['FIPS'].first().apply(lambda x: int(x/1000)).rename('STATEFP').reset_index()
    gdfs[2016]['STATEFP'] = gdfs[2016]['STATEFP'].apply(int)
    gdfs[2016] = gdfs[2016].merge(fips2state, on='STATEFP')
    gdfs[2016]['State'] = gdfs[2016]['State'].apply(lambda x: states_abbr[x])
    gdfs[2016]['CD'] = gdfs[2016]['State'] + "-" + gdfs[2016]['CD115FP'].replace({"00" : 'ALL'})
    logger.info(f"Reading 2018")
    gdfs[2018] = gpd.read_file(Path(config['db_dir']) / "GIS" / "cds_116.shp").to_crs("WGS 84")
    logger.info(f"Reading 2020")
    gdfs[2020] = gpd.read_file(Path(config['db_dir']) / "GIS" / "USA_117th_Congressional_Districts.shp").to_crs("WGS 84")
    gdfs[2020]['CD'] = gdfs[2020]['STATE_ABBR'] + "-" + gdfs[2020]['CDFIPS'].replace({'00' : 'ALL'})
    logger.info(f"Reading 2022")
    gdfs[2022] = gpd.read_file(Path(config['db_dir']) / "GIS" / "118_congress.shp").to_crs("WGS 84")
    gdfs[2022]['CD'] = gdfs[2022]['name'].str.split(" ").str[0]
    gdfs[2022]['CD'] = gdfs[2022]['CD'].str[:2]+"-"+gdfs[2022]['CD'].str[2:].replace({"00" : "ALL"})
    gdf2 = pd.concat([v[['CD', 'geometry']].assign(year=k) for k, v in gdfs.items()])


    logger.info("Caluclating intersecting geometry")
    df = df.join(gdf1[['geometry']], on='FIPS', how='left').merge(gdf2[['geometry', 'CD', 'year']], on=['CD', 'year'], how='left', suffixes=['_x','_y'])
    df = df[df['geometry_x'].notnull() & df['geometry_y'].notnull()]
    df = df[df['Votes'].notnull()]
    t = {}
    for i, row in df.iterrows():
        if i % 100 == 0:
            logger.info(f"{i}/{len(df)}")
        try:
            t[i] = row['geometry_x'].intersection(row['geometry_y'])
        except TopologicalError:
            continue

    df['geometry'] = pd.Series(t)
    df = df[df.geometry.notnull()]
    logger.info("Cutting columns")
    df = df[['CD', 'FIPS', 'wasted_votes', 'wasted_votes_pct', 'excess_votes', 'excess_votes_pct', 'geometry', 'R_votes', 'D_votes', 'Votes', 'year']]
    df = df.merge(process_fips_to_name(), on='FIPS', how='left')
    def filter_polygons(geometry_collection):
        if geometry_collection.type != 'GeometryCollection':
            return geometry_collection
        polygons = []
        for geom in geometry_collection.geoms:
            if isinstance(geom, Polygon):
                polygons.append(geom)
    
        return MultiPolygon(polygons)
    df['geometry'] = df['geometry'].apply(filter_polygons)
    df['geometry'] = gpd.GeoSeries(df['geometry'], crs="EPSG:4326").to_crs('+proj=cea').simplify(1000).to_crs("EPSG:4326")
    df['id'] = df.index
    df = df[df['geometry'] != Polygon()]
    df = df.join(df['geometry'].apply(lambda x: pd.Series(x.centroid.coords[0], index=['long', 'lat'])))
    return df

def main():
    
    house_and_president_by_cd_and_county_list = [[], []]
    for year in range(2016, 2023, 2):
        logger.info(f"Working on {year}")
        house_and_president_by_cd_and_county = read_data(year, include_uncontested=True)
        for i in range(2):
            by_cd_and_county = house_and_president_by_cd_and_county[i]
            if by_cd_and_county is None:
                continue
            by_cd_and_county_list = house_and_president_by_cd_and_county_list[i]
            by_cd = by_cd_and_county.groupby('CD').agg({'votes_margin' : 'sum',  'Votes' : 'sum'}).reset_index()
            by_cd['R-D'] = by_cd['votes_margin']/by_cd['Votes']
            votes_pct = pd.DataFrame({'R_excess_votes_pct' : by_cd['R-D'].apply(lambda x: 2*x/(x+1) if x > 0 else 0),
                                    'D_excess_votes_pct' : by_cd['R-D'].apply(lambda x: 2*x/(x-1) if x < 0 else 0),
                                    'R_wasted_votes_pct' : by_cd['R-D'].apply(lambda x: max(-np.sign(x), 0)),
                                    'D_wasted_votes_pct' : by_cd['R-D'].apply(lambda x: max(np.sign(x), 0)),
                                'CD' : by_cd['CD']})
            by_cd_and_county = by_cd_and_county.merge(votes_pct, on='CD', how='outer')
            by_cd_and_county['R_votes'] = (by_cd_and_county['Votes'] + by_cd_and_county['votes_margin'])/2
            by_cd_and_county['D_votes'] = (by_cd_and_county['Votes'] - by_cd_and_county['votes_margin'])/2
            for var in ['excess_votes', 'wasted_votes']:    
                for i, party in enumerate(['R','D']):
                    by_cd_and_county[f'{party}_{var}'] = by_cd_and_county[f'{party}_{var}_pct']*by_cd_and_county[f'{party}_votes']
                by_cd_and_county[var] = by_cd_and_county[f'R_{var}'] - by_cd_and_county[f'D_{var}']
                by_cd_and_county[f'{var}_pct'] = by_cd_and_county[f'{var}']/by_cd_and_county['Votes']
            
            by_cd_and_county_list += [by_cd_and_county.assign(year=year)]
        
    for name, l in {'house' : house_and_president_by_cd_and_county_list[0],
                    'president' : house_and_president_by_cd_and_county_list[1]}.items():
        fn = Path(config['db_dir']) / f'gerrymandering_{name}.csv'
        logger.info(f"Writing file to {fn}")
        by_cd_and_county = pd.concat(l)
        by_cd_and_county.to_csv(fn, index=False)
        df = add_geometry(by_cd_and_county)
        fn = Path(config['db_dir']) / f'gerrymandering_{name}.geojson'
        logger.info(f"Writing file to {fn}")
        df.to_csv(fn, index=False)
        gpd.GeoDataFrame(df).to_file(fn, driver='GeoJSON')


if __name__ == "__main__":
    main()



                
