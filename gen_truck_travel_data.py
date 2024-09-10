import math
import os
import csv
import time
import graph_gen as gg
import taxicab_st as ts
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from geopy.distance import geodesic

def gen_truck_travel_times():

    problems = ['20170606T113038113409']
    # Use this if all results are desired
    # problems = os.listdir('Problems')

    for problem in problems:
        # Load customer locations
        customers = pd.read_csv(f'Problems/{problem}/tbl_locations.csv')
        customers.columns = customers.columns.str.strip()
        customer_latlons = customers[['latDeg', 'lonDeg']].to_numpy().tolist()
        # 4 km border for the map is sufficient
        lims = get_map_lims(customer_latlons, 4)
        try:
            city = gg.get_city_from_bbox(lims[0], lims[1], lims[2], lims[3])
        except gg.utils.CityNotFoundError:
            print("Customers are in an unknown location")

        # Change directory to graph folder
        try:
            graphfolder = '/graphs'
            os.chdir(os.getcwd() + graphfolder)
        except:
            raise Exception('Graphs folder not found')
        
        # Check if the file exists in the specified folder
        if f'{city}.graphml' not in os.listdir():
            # Graph does not exist, create new one
            gg.generate_graph(lims[0], lims[1], lims[2], lims[3])

        G = ox.load_graphml(filepath=f'{city}.graphml',
                                edge_dtypes={'osmid': str_interpret,
                                            'reversed': str_interpret})

        #_____

        # plot_graph(G, highlight_nodes=[9029518648, 53185265])
        #____

        # Navigate back to original folder         
        os.chdir(os.getcwd().rsplit(graphfolder, 1)[0])

        # Define the path and filename for the CSV
        output_path = f'Problems/{problem}/tbl_truck_travel_data_ST.csv'

        # Open the CSV file for writing
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(['% from location i', 'to location j', 'time [sec]', 'distance [meters]'])

            # Iterate over the customer nodeIDs and write the data rows
            for i in customers['% nodeID']:
                for j in customers['% nodeID']:
                    print(i,j)
                    if i == 0 and j == 10:
                        continue
                    i_pos = (customers.loc[i]['latDeg'], customers.loc[i]['lonDeg'])
                    j_pos = (customers.loc[j]['latDeg'], customers.loc[j]['lonDeg'])
                    time = ts.time.shortest_path(G, i_pos, j_pos)[0]
                    dist = geodesic(i_pos, j_pos).meters
                    writer.writerow([i, j, time, dist])


def plot_graph(G, highlight_nodes=None, depots=False, locs=None, depots_locs=None):
    """Plots the graph of the selected gpkg file and highlights specified nodes."""
    # Plot city graph
    fig, ax = ox.plot_graph(G, show=False, close=False)
    
    # Plot the specified nodes
    if highlight_nodes is not None:
        highlight_positions = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in highlight_nodes}
        highlight_scatter = ax.scatter(
            [pos[0] for pos in highlight_positions.values()],
            [pos[1] for pos in highlight_positions.values()],
            c='red', s=50, zorder=5, label='Highlighted Nodes'
        )
    
    # Plot the customer locations if provided
    if locs is not None:
        customer_scatter = ax.scatter(
            [point.x for _, point in locs.items()],
            [point.y for _, point in locs.items()],
            c='red', s=50, zorder=5, label='Customers'
        )
    
    # Plot the depots if specified
    if depots and depots_locs is not None:
        plural = 's' if len(depots_locs) > 1 else ''
        depot_scatter = ax.scatter(
            [point.x for _, point in depots_locs.items()],
            [point.y for _, point in depots_locs.items()],
            c='blue', s=100, zorder=5, label='Depot' + plural
        )
    
    # Show the plot with a legend
    handles = []
    if highlight_nodes is not None:
        handles.append(highlight_scatter)
    if locs is not None:
        handles.append(customer_scatter)
    if depots and depots_locs is not None:
        handles.append(depot_scatter)
    
    ax.legend(handles=handles)
    plt.show()


def get_map_lims(customer_locs, margin, unit='km'):
    """Function to get map limits where all customers fit in.
    Args: type, description
    margin: float or int, margin for borders of the map
    unit: string, unit for provided margin"""
    
    # Conversion factors
    unit_conversion = {
        'km': 1,
        'm': 1 / 1000,             # 1000 meters in a kilometer
        'mi': 1.60934,             # 1 mile is approximately 1.60934 kilometers
        'nm': 1.852                # 1 nautical mile is approximately 1.852 kilometers
    }

    # Convert margin to kilometers
    if unit in unit_conversion:
        margin_km = margin * unit_conversion[unit]
    else:
        raise ValueError(f"Unsupported unit: {unit}. Use 'km', 'm', 'mi', or 'nm'.")

    # Extract latitudes and longitudes into separate lists
    latitudes = [loc[0] for loc in customer_locs]
    longitudes = [loc[1] for loc in customer_locs]

    # Find the maximum and minimum values
    latmax = max(latitudes)
    latmin = min(latitudes)
    lonmax = max(longitudes)
    lonmin = min(longitudes)

    # Convert margin from km to degrees
    lat_margin_deg = margin_km / 111.32  # 1 degree latitude is approximately 111.32 km
    avg_lat = (latmax + latmin) / 2
    lon_margin_deg = margin_km / (111.32 * math.cos(math.radians(avg_lat)))  # Adjust longitude margin by latitude

    # Calculate the new limits
    box_latmax = latmax + lat_margin_deg
    box_latmin = latmin - lat_margin_deg
    box_lonmax = lonmax + lon_margin_deg
    box_lonmin = lonmin - lon_margin_deg

    # Return the coordinates as a tuple
    return (box_latmax, box_latmin, box_lonmax, box_lonmin)

def str_interpret(value):
    return value  # Ensure the value remains a string

if __name__ == '__main__':
    # disable caching, reduce clutter
    ox.config(use_cache=False)
    # Start time
    start_time = time.time()

    # run main function
    gen_truck_travel_times()

    # End time
    end_time = time.time()

    # Total time taken
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")