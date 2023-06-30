import os,sys
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

sys.path.append('c://projects//osm-flex/src') 

from overlay import raster_to_vector, zonal_stats, vector_point_query, final_land_use
  
def iso2_to_iso3(iso2):
    """
    Converts a two-letter ISO country code (ISO 3166-1 alpha-2) to a three-letter ISO country code (ISO 3166-1 alpha-3).
    
    Parameters:
        iso2 (str): A two-letter ISO country code.
        
    Returns:
        str: The corresponding three-letter ISO country code.
        
    Raises:
        KeyError: If the provided iso2 code is not found in the dictionary.
        
    Example:
        iso2_to_iso3("NL")  # Returns "NLD"
        iso2_to_iso3("FR")  # Returns "FRA"
    """
    
    dict_ = {
        "NL": "NLD",
        "FR": "FRA",
        "DE": "DEU",
        "SW": "SWE",
        "ES": "ESP",
        "PT": "PRT",
        "IT": "ITA",
        "HR": "HRV",
        "GR": "GRC",
        "MT": "MLT",
        "DK": "DNK",
        "BE": "BEL",
        "PL": "POL",
        "CY": "CYP",
        "FI": "FIN",
        "IE": "IRL",
        "EE": "EST",
        "LV": "LVA",
        "RO": "ROU",
        "BG": "BGR",
        "LT": "LTU"
    }
    
    return dict_[iso2]


def prepare_buildings(nuts2, nuts2_europe, bucco_path, CLC_path, coastal_CLC_path):
    """
    Prepares building data for a given NUTS2 region by performing various spatial operations and data processing.
    
    Parameters:
        nuts2 (str): The NUTS2 code of the region to prepare the building data for.
        nuts2_europe (GeoDataFrame): A GeoDataFrame containing the NUTS2 geometries for Europe.
        bucco_path (str): The path to the directory containing the building data files.
        CLC_path (str): The path to the CLC 2018 file.
        coastal_CLC_path (str): The path to the coastal CLC file.
        
    Returns:
        GeoDataFrame: The prepared building data for the specified NUTS2 region.
        None: If the provided NUTS2 code is not in a coastal country or if no buildings are found within the region.
        
    Raises:
        KeyError: If the ISO 2-letter country code cannot be obtained from the NUTS2 code.
        
    Example:
        nuts2 = 'NL32'
        nuts2_europe = gpd.read_file('nuts2_europe.shp')
        bucco_path = '/path/to/bucco_data/'
        CLC_path = '/path/to/CLC2018.tif'
        coastal_CLC_path = '/path/to/coastal_CLC.parquet'
        prepare_buildings(nuts2, nuts2_europe, bucco_path, CLC_path, coastal_CLC_path)
    """

    # get country iso2
    country_iso2 = nuts2[:2]

    # continue if not in a coastal country
    try:
        iso2_to_iso3(country_iso2)
    except KeyError:
        return None

    # read nuts2 geometry
    nuts2_geom = nuts2_europe.loc[nuts2_europe.NUTS_ID == nuts2].geometry.values[0]

    # load buildings for the country
    gdf_bucco = gpd.read_parquet(os.path.join(bucco_path, '{}_bucco.parquet'.format(iso2_to_iso3(country_iso2))))

    # bounding box clip
    bbox_buildings = gdf_bucco.iloc[gdf_bucco.centroid.clip(nuts2_geom.bounds).index].reset_index(drop=True)

    if len(bbox_buildings) == 0:
        return None

    # prepare geometry to improve speed of intersect
    shapely.prepare(bbox_buildings.geometry.values)

    # exact intersect of nuts2 with buildings
    nuts2_buildings = bbox_buildings.loc[shapely.intersects(bbox_buildings.geometry.values, nuts2_geom)].reset_index(drop=True)

    if len(nuts2_buildings) == 0:
        return None

    # get land use information from CLC 2018
    nuts2_buildings['land_use'] = zonal_stats(nuts2_buildings, CLC_path)

    # read coastal corine land cover layer
    coastal_CLC = gpd.read_parquet(coastal_CLC_path)
    coastal_CLC_tree = shapely.STRtree(coastal_CLC.geometry.values)
    band_data_dict = dict(zip(list(coastal_CLC.index), coastal_CLC['CODE_4_18'].values))

    # get centroids to speed up intersect
    nuts2_buildings['centroid'] = nuts2_buildings.centroid

    # get port values
    tqdm.pandas(desc='obtain port values')
    nuts2_buildings['coastal_land_use'] = nuts2_buildings.progress_apply(lambda x: vector_point_query(x, coastal_CLC_tree, band_data_dict), axis=1)

    # get unique use type per building
    tqdm.pandas(desc='get unique use type')
    nuts2_buildings['use_type'] = nuts2_buildings.progress_apply(lambda x: final_land_use(x), axis=1)

    nuts2_buildings = nuts2_buildings.drop(['centroid', 'land_use', 'coastal_land_use'], axis=1)

    return nuts2_buildings

def prepare_cis(nuts2, nuts2_europe, osm_path):
    """
    Prepares critical infrastructure data for a given NUTS2 region by performing various spatial operations and data processing.
    
    Parameters:
        nuts2 (str): The NUTS2 code of the region to prepare the critical infrastructure data for.
        nuts2_europe (GeoDataFrame): A GeoDataFrame containing the NUTS2 geometries for Europe.
        osm_path (str): The path to the directory containing the critical infrastructure data files.
        
    Returns:
        GeoDataFrame: The prepared critical infrastructure data for the specified NUTS2 region.
        None: If the provided NUTS2 code is not in a coastal country.
        
    Raises:
        KeyError: If the ISO 2-letter country code cannot be obtained from the NUTS2 code.
        
    Example:
        nuts2 = 'NL32'
        nuts2_europe = gpd.read_file('nuts2_europe.shp')
        osm_path = '/path/to/osm_data/'
        prepare_cis(nuts2, nuts2_europe, osm_path)
    """

    # get country iso2
    country_iso2 = nuts2[:2]

    # continue if not in a coastal country
    try:
        iso2_to_iso3(country_iso2)
    except KeyError:
        return None

    # read nuts2 geometry
    nuts2_geom = nuts2_europe.loc[nuts2_europe.NUTS_ID == nuts2].geometry.values[0]

    # load osm data for the country
    country_cis = gpd.read_parquet(os.path.join(osm_path, '{}_cis.parquet'.format(iso2_to_iso3(country_iso2))))

    # list of critical infrastructure types
    cis = ['healthcare', 'education', 'gas', 'oil', 'telecom', 'water', 'wastewater', 'power', 'rail', 'road', 'air']

    # loop over critical infrastructure types
    cis_nuts = {}
    for i_cis in cis:
        sub_cis = country_cis.loc[i_cis]
        sub_cis = sub_cis.to_crs(3035)

        # bounding box clip
        bbox_sub_cis = sub_cis.iloc[sub_cis.centroid.clip(nuts2_geom.bounds).index].reset_index(drop=True)

        # prepare geometry to improve speed of intersect
        shapely.prepare(bbox_sub_cis.geometry.values)

        # exact intersect of nuts2 with buildings
        nuts2_cis = bbox_sub_cis.loc[shapely.intersects(bbox_sub_cis.geometry.values, nuts2_geom)].reset_index(drop=True)

        # drop duplicate geometries
        nuts2_cis = nuts2_cis.iloc[nuts2_cis.geometry.to_wkt().drop_duplicates().index].reset_index(drop=True)

        cis_nuts[i_cis] = nuts2_cis

    return gpd.GeoDataFrame(pd.concat(cis_nuts))

def nuts2_exposure(nuts2):
    """
    Generates exposure data for a given NUTS2 region by combining different types of assets and saving the result.

    Parameters:
        nuts2 (str): The NUTS2 code of the region to generate exposure data for.

    Returns:
        GeoDataFrame: The combined exposure data for the specified NUTS2 region.

    Example:
        nuts2 = 'NL32'
        nuts2_exposure(nuts2)
    """

    print('Generating exposure data for {}'.format(nuts2))

    data_path = 'c://data//CEED'
    input_data = os.path.join(data_path, 'input_data')
    bucco_path = os.path.join(data_path, 'coastal_bucco_exact')
    osm_path = os.path.join(data_path, 'coastal_osm_exact')

    # Read NUTS2 data for Europe
    nuts_europe = gpd.read_file(os.path.join(input_data, 'NUTS_RG_03M_2021_3035.shp'))
    nuts2_europe = nuts_europe.loc[nuts_europe.LEVL_CODE == 2].reset_index(drop=True)

    # Load CLC paths
    CLC_path = os.path.join(input_data,'u2018_clc2018_v2020_20u1_raster100m','DATA','U2018_CLC2018_V2020_20u1.tif')
    coastal_CLC_path = os.path.join(input_data,'CZ_2018_DU004_3035_V010.parquet')

    # Prepare building data for the NUTS2 region
    nuts2_buildings = prepare_buildings(nuts2, nuts2_europe, bucco_path, CLC_path, coastal_CLC_path)

    print('INTERMEDIATE UPDATE: Buildings prepared for {}'.format(nuts2))

    # Prepare critical infrastructure data for the NUTS2 region
    nuts2_cis = prepare_cis(nuts2, nuts2_europe, osm_path)

    print('INTERMEDIATE UPDATE: CIS prepared for {}'.format(nuts2))

    all_asset_types = ['buildings', 'healthcare', 'education', 'gas', 'oil', 'telecom', 'water', 'wastewater', 'power', 'rail', 'road', 'air']

    # Combine all asset types into a single dictionary
    combine_all = {}
    for asset_type in all_asset_types:
            
            if asset_type == 'buildings':
                combine_all[asset_type] = nuts2_buildings
            else:
                try:
                    combine_all[asset_type] = nuts2_cis.loc[asset_type]
                except:
                    continue

    # Save the combined exposure data to a file
    out_path = os.path.join(data_path, 'nuts2_CEED', '{}_CEED.parquet'.format(nuts2))
    gpd.GeoDataFrame(pd.concat(combine_all)).to_parquet(out_path)

    return gpd.GeoDataFrame(pd.concat(combine_all))     

if __name__ == "__main__":

    nuts_pilots = ['LT02','ES21','ES52','ITC3','FRI3']
    for nuts2 in nuts_pilots:
        nuts2_combined = nuts2_exposure(nuts2)