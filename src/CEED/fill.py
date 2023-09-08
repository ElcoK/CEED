import os,sys
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import shapely
import pandas as pd
from tqdm import tqdm

sys.path.append('c://projects//osm-flex/src') 

pd.options.mode.chained_assignment = None

def fill_attribute(x,attribute_dict):   
    try:
        return attribute_dict[x]
    except:
        return None
        
def fill_power(country_code):
    """
    Fill the power attribute for towers and poles based on the lines they intersect with.

    Parameters:
        country_code (str): The country code of the country to fill the power attribute for.

    Returns:
        GeoDataFrame or None: The country OSM data with the power attribute filled for towers and poles if the output file doesn't exist,
                                None if the output file already exists.

    Example:
        country_code = 'NL'
        fill_power(country_code)
    """ 

    # Set file paths for data sources and models   
    data_path = 'c://data//CEED'
    osm_path = os.path.join(data_path,'..','CIS_EU')

    # file path:
    out_path = os.path.join(osm_path, '{}_cis.parquet'.format(country_code))

    # Read OSM data for the country
    country_osm = gpd.read_parquet(out_path)   

    # grab power infrastructure data
    power_infra = country_osm.loc['power']
    towers = power_infra.loc[power_infra.power == 'tower']
    poles = power_infra.loc[power_infra.power == 'pole']
    lines = power_infra.loc[power_infra.power == 'line']

    # create a small buffer around the towers and poles to have a larger probability that it will intersect
    towers.geometry = towers.to_crs(3035).buffer(5)
    poles.geometry = poles.to_crs(3035).buffer(5)

    # create spatial tree of the lines
    tree_lines = shapely.STRtree(lines.to_crs(3035).geometry) 

    tqdm.pandas(desc='Intersect lines and towers')
    df_tower = pd.DataFrame(tree_lines.query(towers.geometry,predicate='intersects').T,columns=['tower_id','lines_id'])
    df_tower['voltage'] = df_tower.lines_id.progress_apply(lambda x: lines.iloc[x]['voltage'])

    tqdm.pandas(desc='Intersect lines and poles')  
    df_poles = pd.DataFrame(tree_lines.query(poles.geometry,predicate='intersects').T,columns=['poles_id','lines_id'])
    df_poles['voltage'] = df_poles.lines_id.progress_apply(lambda x: lines.iloc[x]['voltage'])

    tqdm.pandas(desc='Fill tower gaps when possible')
    voltage_dict = df_tower.groupby('tower_id')['voltage'].first().to_dict() 
    towers['voltage'] = towers.progress_apply(lambda x: fill_attribute(x.name,voltage_dict),axis=1) 
            
    tqdm.pandas(desc='Fill pole gaps when possible')    
    voltage_dict = df_poles.groupby('poles_id')['voltage'].first().to_dict() 
    poles['voltage'] = poles.progress_apply(lambda x: fill_attribute(x.name,voltage_dict),axis=1) 

    #put the data back in
    country_osm.loc[country_osm['power'] == 'tower','voltage'] = towers.voltage.values
    country_osm.loc[country_osm['power'] == 'pole','voltage'] = poles.voltage.values

    country_osm.to_parquet(out_path)

    return country_osm


if __name__ == "__main__":

    # Set file paths for data sources and models    
    data_path = 'c://data//CEED'
    osm_path = os.path.join(data_path,'..','CIS_EU')

    country_codes = [x.split('.')[0][:3] for x in os.listdir(osm_path) if x.endswith('.parquet')]

    for country_code in country_codes:
        print(country_code)
        fill_power(country_code)