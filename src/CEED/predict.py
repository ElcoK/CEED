import os,sys
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import shapely
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

sys.path.append('c://projects//osm-flex/src') 


pd.options.mode.chained_assignment = None
def landuse_simplified(value):

    land_use_dict = {
    1 : 1,
    2 : 2,
    3 : 3,
    4 : 4,
    5 : 5,
    6 : 6,
    7 : 7,
    8 : 8,
    9 : 9,
    10 : 10,
    11 : 11,
    12 : 12,
    13 : 12,
    14 : 12,
    15 : 12,
    16 : 13,
    17 : 13,
    18 : 14,
    19 : 15,
    20 : 15,
    21 : 15,
    22 : 16,
    23 : 16,
    24 : 16,
    25 : 16,
    26 : 16,
    27 : 16,
    28 : 16,
    29 : 16,
    30 : 17,
    31 : 17,
    32 : 17,
    33 : 17,
    34 : 17,
    35 : 18,
    36 : 18,
    37 : 18,
    38 : 18,
    39 : 18,
    40 : 19,
    41 : 19,
    42 : 19,
    43 : 19,
    44 : 19,
    48 : 19,    }    
    
    return land_use_dict[value]


def raster_to_vector(xr_raster):
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
    df_1 = df_1.drop(['x', 'y', 'band', 'spatial_ref'], axis=1)

    # Calculate the resolution of the raster
    resolution = xr_raster.x[1].values - xr_raster.x[0].values

    # Buffer the Point geometries by half of the resolution with square caps
    df_1.geometry = shapely.buffer(df_1.geometry, distance=resolution/2, cap_style='square').values

    # Convert the DataFrame to a GeoDataFrame
    return gpd.GeoDataFrame(df_1)      

def zonal_stats(vector, raster_in,zonal_landuse=True):
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
    tqdm.pandas(desc='perform zonal statistics')
    
    raster = raster.rio.write_crs(3035)
    
    # Clip the raster dataset to the bounding box of the vector dataset
    raster_clip = raster.rio.clip_box(vector.total_bounds[0], vector.total_bounds[1], vector.total_bounds[2], vector.total_bounds[3])
       
    
    # Convert the clipped raster dataset to a vector representation
    raster_vector = raster_to_vector(raster_clip)
    
    # Create a dictionary mapping each index to its corresponding band data value
    band_data_dict = dict(zip(list(raster_vector.index), raster_vector['band_data'].values))
    
    get_highest = (raster_vector['band_data'].value_counts().index[0])
    
    # Construct an STRtree from the vector geometry values
    tree = shapely.STRtree(raster_vector.geometry.values)
    
    # Apply a function to calculate zonal statistics for each centroid point in the vector dataset
    
    if zonal_landuse:
        return vector.centroid.progress_apply(lambda x: band_data_dict[tree.query(x, predicate='intersects')[0]] if len(tree.query(x, predicate='intersects')) > 0 else get_highest)
    else:
        return vector.centroid.progress_apply(lambda x: band_data_dict[tree.query(x, predicate='intersects')[0]] if len(tree.query(x, predicate='intersects')) > 0 else 0)
        

def develop_predictor(country_code, data, y_col='maxspeed', x_cols=['population', 'land_use','slope', 'highway', 'lanes', 'surface'], inline=True):
    """
    Trains a random forest classifier model to predict a target variable based on the given input features.
    
    Parameters:
        country_code (str): The country code of the country to train the model for.
        data (pandas.DataFrame): The input data containing both the features and the target variable.
        y_col (str, optional): The name of the target variable column in the data. Default is 'maxspeed'.
        x_cols (list, optional): A list of feature column names in the data. Default is ['land_use', 'highway', 'lanes', 'surface'].
        inline (bool, optional): Determines whether to print the model accuracy inline or return the model pipeline. 
                                If True, the accuracy is printed inline. If False, the model pipeline is returned. Default is True.
    
    Returns:
        sklearn.pipeline.Pipeline or None: If inline is True, the function prints the model accuracy.
                                           If inline is False, the function returns the trained model pipeline.
    """
    
    # Separate the target variable and input features
    y = data[y_col]
    X = data[x_cols]
        
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.66)
    
    # Identify the categorical features for one-hot encoding
    features_to_encode = X_train.columns[X_train.dtypes == object].tolist()
       
    # Create a column transformer to one-hot encode the categorical features and pass through the remaining features
    col_trans = make_column_transformer(
        (OneHotEncoder(), features_to_encode),
        remainder="passthrough"
    )
    
    # Create a random forest classifier with specified parameters
    rf_classifier = RandomForestClassifier(
        criterion='gini',
        min_samples_leaf=25,
        n_estimators=50,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1
    )
    
    # Create a pipeline with the column transformer and random forest classifier
    pipe = make_pipeline(col_trans, rf_classifier)
    
    # Train the model pipeline
    pipe.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = pipe.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test.values.flatten(), y_pred)

    # # Calculate precision score
    precision = precision_score(y_test.values.flatten(), y_pred,average='weighted')

    # # Calculate recall score
    recall = recall_score(y_test.values.flatten(), y_pred,average='weighted')
    
    # # Calculate recall score
    f_score = f1_score(y_test.values.flatten(), y_pred,average='weighted')  
    
    pd.DataFrame([accuracy,precision,recall,f_score],index=['accuracy','precision','recall','f_score'],columns=[f"{country_code}_{y_col}"]).to_csv(f'c://data//CEED//scores//{country_code}_{y_col}_scores.csv',mode='a')

    # Print the accuracy if inline is False
    if not inline:
        print(f"The accuracy of the {y_col} model is {round(accuracy * 100, 3)} %")
        print(f"The precision of the {y_col} model is {round(precision * 100, 3)} %")
        print(f"The recall of the {y_col} model is {round(recall * 100, 3)} %")   
        print(f"The F1-score of the {y_col} model is {round(f_score * 100, 3)} %")   

    # Return the model pipeline if inline is False
    return pipe

def fill_attributes(x,infra_specific_tag,pipe_dict,attributes):
    """
    Fills missing attributes (e.g., maxspeed, lanes, and surface) in a data point using trained prediction models.
    
    Parameters:
        x (pandas.Series): A single road data point containing 'landuse', 'highway', 'sinuosity', 'maxspeed', 'lanes', and 'surface' attributes.
    
    Returns:
        pandas.Series: The updated road data point with filled missing attributes.
    """
    
    # Create a DataFrame with the necessary features for prediction
    if 'sinuosity' in x.index:
        X_pred = pd.DataFrame(x[['population','landuse','slope',infra_specific_tag, 'sinuosity']]).T
    else:
        X_pred = pd.DataFrame(x[['population','landuse','slope',infra_specific_tag]]).T
        

    for attribute in attributes:
        if x[attribute] is None:
            x[attribute] = pipe_dict[attribute].predict(X_pred)[0]
    
    return x

def sinuosity(geom):
    if geom.geom_type == 'MultiPolygon':
        return 1
    elif geom.geom_type == 'LineString':      
        return shapely.length(geom)/shapely.distance(shapely.get_point(geom,0),shapely.get_point(geom,-1))
    
    
def update_country_attributes(country_code, infra_type='road',
                                            infra_specific_tag = ['highway'], 
                                            attributes=['surface','lanes','maxspeed'],
                                            **kwargs):
    """
    Updates road attributes (maxspeed, lanes, and surface) for a specific country based on various data sources and models.
    
    Parameters:
        country_code (str): The country code for the country to update the road attributes.
    
    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing updated road attributes for the coastal areas of the country.
    """
        
        
    # Set file paths for data sources and models   
    data_path = 'c://data//CEED'
    input_data = os.path.join(data_path,'input_data')
    osm_path = os.path.join(data_path,'..','CIS_EU')

    bucco_file = os.path.join(input_data, '..', 'coastal_bucco_exact', '{}_bucco.parquet').format(country_code)
    CLC_path = os.path.join(input_data, 'u2018_clc2018_v2020_20u1_raster100m', 'DATA', 'U2018_CLC2018_V2020_20u1.tif')
    pop_path = os.path.join(input_data, 'ppp_2018_1km_EU.tif')
    slope_path = os.path.join(input_data, 'eudem_slop_3035_europe_1km.tif')

    # Read OSM data for the country
    country_osm = gpd.read_parquet(os.path.join(osm_path, '{}_cis.parquet'.format(country_code)))   

    # Extract relevant road attributes and convert to the desired coordinate reference system
    if kwargs is None:
        infrastructure = gpd.GeoDataFrame(country_osm.loc[infra_type][['geometry']+infra_specific_tag+attributes])
    elif kwargs['geom_type'] in ['LineString','Point']:
        infrastructure = gpd.GeoDataFrame(country_osm.loc[infra_type][['geometry']+infra_specific_tag+attributes])
        infrastructure = infrastructure.loc[infrastructure.geometry.geom_type == kwargs['geom_type']]

    infrastructure = infrastructure.to_crs(3035)

    # remove links from roads
    if infra_type == 'road':
        infrastructure[infra_specific_tag[0]] = infrastructure[infra_specific_tag[0]].str.rsplit(pat="_",expand=True, n=0)[0]

    if infra_type in ['rail','power']:
        infrastructure = infrastructure.loc[infrastructure[infra_specific_tag[0]].isin(kwargs['asset_types'])]

    # Perform zonal statistics to extract land use information for the roads
    land_use = zonal_stats(infrastructure, CLC_path)
    infrastructure['landuse'] = land_use
    infrastructure['landuse'] = infrastructure['landuse'].apply(lambda x: landuse_simplified(x))

    slope = zonal_stats(infrastructure, slope_path,zonal_landuse=False)
    infrastructure['slope'] = slope

    population = zonal_stats(infrastructure, pop_path,zonal_landuse=False)
    infrastructure['population'] = population

    # Calculate the sinuosity of each road and cap extreme values
    if kwargs['geom_type'] == 'LineString':
        tqdm.pandas(desc='obtain sinuosity')
        infrastructure['sinuosity'] = infrastructure.geometry.progress_apply(lambda x: sinuosity(x))
        infrastructure.sinuosity.loc[infrastructure.sinuosity > infrastructure.sinuosity.quantile(.98)] = infrastructure.sinuosity.quantile(.98)

    # Drop rows with missing values
    full_data = infrastructure.dropna()

    if len(full_data) == 0:
        return None
    
    full_data = full_data[full_data[infra_specific_tag[0]].map(full_data[infra_specific_tag[0]].value_counts()) > 10]

    # Update infrequent surface values to the most common type
    for attribute in attributes:
        full_data.loc[full_data[attribute].map(full_data[attribute].value_counts(normalize=True).lt(0.01)), attribute] = full_data[attribute].value_counts().index[0]

    # Update infrequent landuse values with the minimum unique value
    full_data.loc[full_data['landuse'].map(full_data['landuse'].value_counts(normalize=True).lt(0.01)), 'landuse'] = full_data['landuse'].value_counts().index[0]

    # Convert landuse column to object type
    full_data.landuse = full_data.landuse.astype('object')

    # Develop predictor models for lanes, maxspeed, and surface
    pipe_dict = {}
    for attribute in attributes:
        if kwargs['geom_type'] == 'LineString':
            pipe_dict[attribute] = develop_predictor(country_code,full_data, y_col=attribute, x_cols=['population','landuse','slope', infra_specific_tag[0], 'sinuosity'], inline=False)
        else:
            pipe_dict[attribute] = develop_predictor(country_code,full_data, y_col=attribute, x_cols=['population','landuse','slope', infra_specific_tag[0]], inline=False)
            

    # Set paths for input and output coastal OSM data
    coastal_path = os.path.join(data_path, 'coastal_osm_exact')

    # Read coastal OSM data for the country
    coastal_osm = gpd.read_parquet(os.path.join(coastal_path, '{}_cis.parquet'.format(country_code)))

    if kwargs is None:
        coastal_infra = gpd.GeoDataFrame(coastal_osm.loc[infra_type][['geometry']+infra_specific_tag+attributes])
    elif kwargs['geom_type'] in ['LineString','Point']:
        coastal_infra = gpd.GeoDataFrame(coastal_osm.loc[infra_type][['geometry']+infra_specific_tag+attributes])       
        coastal_infra = coastal_infra.loc[coastal_infra.geometry.geom_type == kwargs['geom_type']]

    coastal_infra = coastal_infra.to_crs(3035)

    # remove 'links from highway'
    if infra_type == 'road':
        coastal_infra[infra_specific_tag[0]] = coastal_infra[infra_specific_tag[0]].str.rsplit(pat="_",expand=True, n=0)[0]

    if infra_type in ['rail','power']:
        coastal_infra = coastal_infra.loc[coastal_infra[infra_specific_tag[0]].isin(kwargs['asset_types'])]
        
    # Perform zonal statistics to extract land use information for the coastal roads
    coastal_infra['landuse'] = zonal_stats(coastal_infra, CLC_path)
    coastal_infra['landuse'] = coastal_infra['landuse'].apply(lambda x: landuse_simplified(x))
    coastal_infra.landuse = coastal_infra.landuse.astype('object')

    coastal_infra['population'] = zonal_stats(coastal_infra, pop_path,zonal_landuse=False)
    coastal_infra['slope'] = zonal_stats(coastal_infra, slope_path,zonal_landuse=False)

    # Calculate the sinuosity of each coastal road and cap extreme values
    if kwargs['geom_type'] == 'LineString':
        coastal_infra['sinuosity'] = coastal_infra.geometry.apply(lambda x: sinuosity(x))
        coastal_infra.sinuosity.loc[coastal_infra.sinuosity > coastal_infra.sinuosity.quantile(.98)] = coastal_infra.sinuosity.quantile(.98)

    # Update infrequent highway values as the most common
    coastal_infra.loc[coastal_infra[infra_specific_tag[0]].map(coastal_infra[infra_specific_tag[0]].value_counts(normalize=True).lt(0.005)), 
                    infra_specific_tag[0]] = coastal_infra[infra_specific_tag[0]].value_counts().index[0]

    # Update infrequent landuse values with the most common value
    coastal_infra.loc[coastal_infra['landuse'].map(coastal_infra['landuse'].value_counts(normalize=True).lt(0.1)), 
                    'landuse'] = coastal_infra.landuse.value_counts().index[0]

    coastal_infra.landuse = coastal_infra.landuse.astype('object')

    # Fill missing road attributes for coastal roads using the fill_road_attributes function
    tqdm.pandas(desc='fill missing values')
    coastal_infra = coastal_infra.progress_apply(lambda x: fill_attributes(x,infra_specific_tag[0],pipe_dict,attributes), axis=1)

    # Update road attributes in the coastal OSM data with the filled values
    for attribute in attributes:
        if kwargs is None:
            coastal_osm.loc[infra_type, attribute] = coastal_infra[attribute].values
        elif (kwargs['geom_type'] in ['LineString','Point']) & (len(kwargs) == 2):
            coastal_osm.loc[(coastal_osm.index.get_level_values(0) == infra_type) & 
                (coastal_osm.geometry.geom_type == kwargs['geom_type']) & 
                    (coastal_osm[infra_specific_tag].isin(kwargs['asset_types'])).values.flatten(),attribute] = coastal_infra[attribute].values
        elif (kwargs['geom_type'] in ['LineString','Point']) & (len(kwargs) == 1):
           coastal_osm.loc[(coastal_osm.index.get_level_values(0) == infra_type) & 
                (coastal_osm.geometry.geom_type == kwargs['geom_type']),attribute] = coastal_infra[attribute].values

    coastal_osm.to_parquet(os.path.join(coastal_path, '{}_cis.parquet'.format(country_code)))
    return coastal_osm

if __name__ == "__main__":

    # Set file paths for data sources and models    
    data_path = 'c://data//CEED'
    osm_path = os.path.join(data_path,'..','CIS_EU')

    country_codes = [x.split('.')[0][:3] for x in os.listdir(osm_path) if x.endswith('.parquet')]
    
    all_files = [os.path.join(osm_path, '{}_cis.parquet'.format(country_code)) for country_code in country_codes]
    sorted_files = sorted(all_files, key = os.path.getsize) 
    
    country_codes = [x.split('\\')[3][:3] for x in sorted_files if x.endswith('.parquet')]
    
    for country_code in country_codes:
        try:
            predict_input_dict = {
                # 'road' : (['highway'],['surface','lanes','maxspeed'],'LineString',),    
                # 'rail' : (['railway'],['gauge','electrified','voltage'],'LineString',['rail','light_rail']),
            #    'power_line' : (['power'],['voltage'],'LineString',['line','minor_line']),
                'power_point' : (['power'],['voltage'],'Point',['pole','tower','substation','transformer']),  
            }

            for input_ in predict_input_dict.keys():
                infra_specific_dict = list(predict_input_dict[input_])
                infra_type = input_.split('_')[0]        
                infra_specific_tag = infra_specific_dict[0]
                attributes = infra_specific_dict[1]
                geom_type = infra_specific_dict[2]
                
                print(country_code,infra_type)

                if len(infra_specific_dict) > 3:
                    asset_types = infra_specific_dict[3]

                    update_country_attributes(country_code, infra_type=infra_type,
                                                            infra_specific_tag = infra_specific_tag, 
                                                            attributes=attributes,geom_type=geom_type,asset_types=asset_types)
                    
                else:
                    update_country_attributes(country_code, infra_type=infra_type,
                                                            infra_specific_tag = infra_specific_tag, 
                                                           attributes=attributes,geom_type=geom_type)            
        except:
            print(f"{country_code} failed")
            continue

