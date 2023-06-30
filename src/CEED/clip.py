import os,sys
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import shapely
import pandas as pd
import xarray as xr
from tqdm import tqdm


def set_paths():
    # set paths
    data_path = 'c://data//CEED'
    input_data = os.path.join(data_path,'input_data')
    cis_data = os.path.join(data_path,'..','CIS_EU')
    bucco_path = os.path.join(data_path,'..','EUBUCCO')

    return data_path,input_data,cis_data,bucco_path

def read_coastal_mask(input_data, crs=3035, set_crs=3035):
    """
    Reads the coastal mask data from a Parquet file and returns it as a GeoDataFrame.

    Parameters:
        input_data (str): The path to the directory containing the coastal mask Parquet file.
        crs (int or str, optional): The target coordinate reference system (CRS) of the returned GeoDataFrame.
            Defaults to EPSG:3035.

    Returns:
        GeoDataFrame: The coastal mask data as a GeoDataFrame.

    Example:
        input_data = '/path/to/input_data/'
        crs = 3035
        read_coastal_mask(input_data, crs)
    """

    mask = gpd.read_parquet(os.path.join(input_data, 'coastal_mask.parquet'))
    mask = mask.set_crs(crs)

    if set_crs == 3035:
        return mask
    else:
        return mask.to_crs(set_crs)

def clip_bucco(country_code):
    """
    Clips the coastal bucco data for a specified country to the coastal mask and saves the clipped data.

    Parameters:
        country_code (str): The country code of the country to clip the bucco data for.

    Returns:
        GeoDataFrame or None: The clipped coastal bucco data as a GeoDataFrame if the output file doesn't exist,
                              None if the output file already exists.

    Example:
        country_code = 'NL'
        clip_bucco(country_code)
    """

    # Load paths
    data_path, input_data, cis_data, bucco_path = set_paths()

    # Set file paths
    file_path = os.path.join(bucco_path, '{}_bucco.parquet'.format(country_code))
    out_path = os.path.join(input_data, '..', 'coastal_bucco', '{}_bucco.parquet').format(country_code)

    # Load coastal mask
    mask = read_coastal_mask(input_data)

    # Check if the output file already exists
    if os.path.exists(out_path):
        return None
    else:
        bucco = gpd.read_parquet(file_path)
        coastal_bucco = bucco.iloc[bucco.centroid.clip(mask).index].reset_index(drop=True)
        coastal_bucco.to_parquet(out_path)

        return coastal_bucco
    
def all_bucco():
    """
    Clips the coastal bucco data for all countries to the coastal mask.

    Returns:
        None

    Example:
        all_bucco()
    """

    # Load paths
    data_path, input_data, osm_data, cis_data, bucco_path = set_paths()

    # Get country codes
    country_codes = [x.split('.')[0][-3:] for x in os.listdir(osm_data) if x.endswith('.pbf')]

    # Clip bucco data for each country
    for country_code in tqdm(country_codes, total=len(country_codes)):
        clip_bucco(country_code)


def clip_cis(country_code):
    """
    Clips the coastal cis (critical infrastructure) data for a specified country to the coastal mask and saves the clipped data.

    Parameters:
        country_code (str): The country code of the country to clip the cis data for.

    Returns:
        GeoDataFrame or None: The clipped coastal cis data as a GeoDataFrame if the output file doesn't exist,
                              None if the output file already exists.

    Example:
        country_code = 'NL'
        clip_cis(country_code)
    """

    # Load paths
    data_path, input_data, cis_data, bucco_path = set_paths()

    # Set file paths
    file_path = os.path.join(cis_data, '{}_cis.parquet'.format(country_code))
    out_path = os.path.join(input_data, '..', 'coastal_osm', '{}_cis.parquet').format(country_code)

    # Load coastal mask
    mask = read_coastal_mask(input_data, set_crs=4326)

    # Check if the output file already exists
    if os.path.exists(out_path):
        return None
    else:
        osm = gpd.read_parquet(file_path)
        collect_all = {}
        for infra in osm.groupby(level=0):
            uniq_infra = infra[1].reset_index(drop=True)
            collect_all[infra[0]] = uniq_infra.loc[uniq_infra.centroid.clip(mask).index].reset_index(drop=True)

        cis_country = gpd.GeoDataFrame(pd.concat(collect_all))

        cis_country.to_parquet(out_path)

        return cis_country

    
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
        clip_cis(country_code)

    
if __name__ == "__main__":
    all_cis_osm()