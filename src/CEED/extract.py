import os,sys
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
from tqdm import tqdm 

sys.path.append('c://projects//osm-flex/src') 
import osm_flex.extract as ex

def set_paths():
    # set paths
    data_path = 'c://data//CEED'
    input_data = os.path.join(data_path,'input_data')
    osm_data = os.path.join(data_path,'..','country_osm')
    cis_data = os.path.join(data_path,'..','CIS_EU')
    bucco_path = os.path.join(data_path,'..','EUBUCCO')

    return data_path,input_data,osm_data,cis_data,bucco_path

def country_cis_osm(country_code):
    """
    Process critical infrastructure systems (CIS) data for a specific country using OpenStreetMap (OSM) data.

    Parameters:
    - country_code (str): The country code used to identify the country.

    Returns:
    - None

    Side Effects:
    - Generates and saves the CIS data as a Parquet file for the specified country.

    """

    # List of critical infrastructure systems to process
    cis = ['healthcare', 'education', 'gas', 'oil', 'telecom', 'water', 'wastewater', 'power', 'rail', 'road', 'air']

    # Load paths
    data_path, input_data, osm_data, cis_data, bucco_path = set_paths()

    # Set paths
    country_pbf = os.path.join(osm_data, '{}.osm.pbf'.format(country_code))
    out_path = os.path.join(cis_data, '{}_cis.parquet').format(country_code)

    # Check if the output file already exists
    if os.path.exists(out_path):
        return None
    else:
        # Extract CIS
        collect_cis = {}
        for i_cis in cis:
            print(i_cis, country_code)
            collect_cis[i_cis] = ex.extract_cis(country_pbf, i_cis)

        # Save the extracted CIS data to Parquet format
        gpd.GeoDataFrame(pd.concat(collect_cis)).to_parquet(out_path)

def all_cis_osm():
    """
    Process critical infrastructure systems (CIS) data for all available countries using OpenStreetMap (OSM) data.

    Returns:
    - None

    Side Effects:
    - Generates and saves the CIS data as Parquet files for each country.

    """

    # Load paths
    data_path, input_data, osm_data, cis_data, bucco_path = set_paths()

    # Grab country codes from OSM data directory
    country_codes = [x.split('.')[0][-3:] for x in os.listdir(osm_data) if x.endswith('.pbf')]

    # Process CIS for each country
    for country_code in tqdm(country_codes, total=len(country_codes)):
        country_cis_osm(country_code)


def country_bucco(country_code):
    """
    Process BUCCO (Building Use, Cover, and Complexity Observatory) data for a specific country.

    Parameters:
    - country_code (str): The country code used to identify the country.

    Returns:
    - None

    Side Effects:
    - Generates and saves the BUCCO data as a Parquet file for the specified country.

    """

    # Load paths
    data_path, input_data, osm_data, cis_data, bucco_path = set_paths()

    file_path = os.path.join(bucco_path, 'v0_1-{}.gpkg'.format(country_code))
    out_path = os.path.join(bucco_path, '{}_bucco.parquet').format(country_code)

    # Check if the output file already exists
    if os.path.exists(out_path):
        return None
    else:
        # Read the BUCCO data from the file
        bucco = gpd.read_file(file_path)
        print('BUCCO loaded')

        # Save the BUCCO data to Parquet format
        bucco.to_parquet(out_path)


def all_bucco():

    # load paths
    data_path,input_data,osm_data,cis_data,bucco_path = set_paths()

    # grab country codes
    country_codes = [x.split('.')[0][-3:] for x in os.listdir(osm_data) if x.endswith('.pbf')]

    for country_code in tqdm(country_codes,total=len(country_codes)):
        country_bucco(country_code)

if __name__ == "__main__":
    all_cis_osm()