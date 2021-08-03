'''
https://developers.google.com/earth-engine/guides 
'''

# Earth Engine Python API
import pprint
import ee
ee.Authenticate()
ee.Initialize()

import folium


def add_ee_layer(self, ee_image_object, vis_params, name):
  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
  folium.raster_layers.TileLayer(
      tiles=map_id_dict['tile_fetcher'].url_format,
      attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
      name=name,
      overlay=True,
      control=True
  ).add_to(self)

folium.Map.add_ee_layer = add_ee_layer

# Define a point in the area where we want to acquire an image (Des Moines, Iowa)
area_of_interest = ee.Geometry.Point(-93.6765557, 41.5666487)

tile = ee.ImageCollection('COPERNICUS/S2_SR')\
.filterBounds(area_of_interest)\
.filterDate('2020-01-01', '2020-12-31')\
.sort('CLOUDY_PIXEL_PERCENTAGE')\
.first()

rgb_tile = tile.visualize(bands = ['B4', 'B3', 'B2'], max = 4000)
pp = pprint.PrettyPrinter()
pp.pprint(rgb_tile)

# Load an image.
image = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318')

# Define the visualization parameters.
image_viz_params = {
    'bands': ['B5', 'B4', 'B3'],
    'min': 0,
    'max': 0.5,
    'gamma': [0.95, 1.1, 1]
}

# Define a map centered on San Francisco Bay.
map_l8 = folium.Map(location=[37.5010, -122.1899], zoom_start=10)

# Add the image layer to the map and display it.
map_l8.add_ee_layer(image, image_viz_params, 'false color composite')
display(map_l8)