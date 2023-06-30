import os,sys
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
sys.path.append('c://projects//osm-flex/src') 


def raster_to_vector(xr_raster,band_column='band'):
    """
    Convert a raster to a vector representation.

    Args:
        xr_raster (xarray.DataArray): Input raster data as xarray.DataArray.

    Returns:
        gpd.GeoDataFrame: Vector representation of the input raster.
    """

    # Convert xarray raster to pandas DataFrame
    df = xr_raster.to_dataframe()

    # Filter DataFrame to select rows where band_data is 1
    df_1 = df.loc[df.band_data > 0].reset_index()

    # Create a Shapely Point geometry column from x and y values
    df_1['geometry'] = shapely.points(df_1.x.values, df_1.y.values)

    # Remove unnecessary columns from the DataFrame
    df_1 = df_1.drop(['x', 'y', band_column, 'spatial_ref'], axis=1)

    # Calculate the resolution of the raster
    resolution = xr_raster.x[1].values - xr_raster.x[0].values

    # Buffer the Point geometries by half of the resolution with square caps
    df_1.geometry = shapely.buffer(df_1.geometry, distance=resolution/2, cap_style='square').values

    # Convert the DataFrame to a GeoDataFrame
    return gpd.GeoDataFrame(df_1)      

def zonal_stats(vector, raster_in):
    """
    Calculate zonal statistics of a raster dataset based on a vector dataset.
    
    Parameters:
    - vector_in (str): Path to the vector dataset file (in Parquet format).
    - raster_in (str): Path to the raster dataset file (in NetCDF format).
    
    Returns:
    - pandas.Series: A series containing the zonal statistics values corresponding to each centroid point in the vector dataset.
    """
    
    # Open the raster dataset using the xarray library
    raster = xr.open_dataset(raster_in, engine="rasterio")
    
    # Progress bar setup for obtaining values
    tqdm.pandas(desc='obtain values')
    
    # Clip the raster dataset to the bounding box of the vector dataset
    raster_clip = raster.rio.clip_box(vector.total_bounds[0], vector.total_bounds[1], vector.total_bounds[2], vector.total_bounds[3])
    
    # Convert the clipped raster dataset to a vector representation
    raster_vector = raster_to_vector(raster_clip)
    
    # Create a dictionary mapping each index to its corresponding band data value
    band_data_dict = dict(zip(list(raster_vector.index), raster_vector['band_data'].values))
    
    # Construct an STRtree from the vector geometry values
    tree = shapely.STRtree(raster_vector.geometry.values)
    
    # Apply a function to calculate zonal statistics for each centroid point in the vector dataset
    return vector.centroid.progress_apply(lambda x: band_data_dict[tree.query(x, predicate='intersects')[0]])


def vector_point_query(x, coastal_CLC_tree, band_data_dict):
    """
    Perform a point query on a vector dataset based on specific conditions.

    Parameters:
    - x (GeoDataFrame): A GeoDataFrame representing a single feature.
    - coastal_CLC_tree (shapely.STRtree): STRtree object constructed from the coastal CLC vector geometry values.
    - band_data_dict (dict): A dictionary mapping indices to their corresponding band data values.

    Returns:
    - int: The band data value corresponding to the point query, or -9999 if the conditions are not met.
    """

    if x.land_use == 5:
        try:
            # Perform an intersection query using the centroid of the feature
            match = coastal_CLC_tree.query(x.centroid, predicate='intersects')
            return band_data_dict[match[0]]
        except:
            # Return -9999 if no intersection is found
            return -9999
    else:
        # Return -9999 if the land use condition is not met
        return -9999
    
def final_land_use(x):
    """
    Determine the final land use based on the coastal land use and land use values of a feature.

    Parameters:
    - x (pandas.Series): A pandas Series representing a single feature.

    Returns:
    - int: The final land use value, which is either the coastal land use or the land use value of the feature.
    """

    if x.coastal_land_use == -9999:
        return x.land_use
    else:
        return x.coastal_land_use  