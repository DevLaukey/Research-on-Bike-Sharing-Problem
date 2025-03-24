import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from scipy.spatial import distance
import networkx as nx
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate simulated bike stations in Lyon
def generate_lyon_stations(n_stations=150):
    """
    Generate simulated bike stations with random locations and inventory levels.
    """
    # Approximate coordinates for Lyon city center
    lyon_center = [4.835, 45.76]  
    
    # Generate station locations with a normal distribution around city center
    longs = np.random.normal(lyon_center[0], 0.03, n_stations)
    lats = np.random.normal(lyon_center[1], 0.02, n_stations)
    
    # Generate random current and target inventory levels
    current_inventory = np.random.randint(0, 20, n_stations)
    target_inventory = np.random.randint(5, 20, n_stations)
    
    # Calculate imbalance (positive means surplus, negative means deficit)
    imbalance = current_inventory - target_inventory
    
    # Create DataFrame with station data
    stations = pd.DataFrame({
        'station_id': range(1, n_stations + 1),
        'longitude': longs,
        'latitude': lats,
        'current_inventory': current_inventory,
        'target_inventory': target_inventory,
        'imbalance': imbalance
    })
    
    return stations

# Function to cluster stations
def cluster_stations(stations, n_clusters=8):
    """
    Cluster bike stations into groups using KMeans clustering.
    """
    coordinates = stations[['longitude', 'latitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    stations['cluster'] = kmeans.fit_predict(coordinates)
    
    return stations, kmeans.cluster_centers_

# Function to identify stations needing rebalancing
def identify_rebalancing_needs(stations, threshold=3):
    """
    Identify stations with surplus or deficit of bikes beyond a given threshold.
    """
    surplus_stations = stations[stations['imbalance'] > threshold].copy()
    surplus_stations['type'] = 'surplus'
    
    deficit_stations = stations[stations['imbalance'] < -threshold].copy()
    deficit_stations['type'] = 'deficit'
    
    balanced_stations = stations[(stations['imbalance'] >= -threshold) & 
                                 (stations['imbalance'] <= threshold)].copy()
    balanced_stations['type'] = 'balanced'
    
    stations_with_rebalancing = pd.concat([surplus_stations, deficit_stations])
    
    return surplus_stations, deficit_stations, balanced_stations, stations_with_rebalancing

# Function to optimize rebalancing routes
def optimize_cluster_routes(stations_with_rebalancing, cluster_centers, with_outliers=True):
    """
    Optimize rebalancing routes within clusters, with or without considering outliers.
    """
    routes = {}
    
    if not with_outliers:
        outlier_threshold = 0.95  # Top 5% are outliers
        
        # Compute distances to cluster centers
        for idx, station in stations_with_rebalancing.iterrows():
            cluster_idx = station['cluster']
            center = cluster_centers[cluster_idx]
            distance_to_center = np.sqrt((station['longitude'] - center[0])**2 + 
                                         (station['latitude'] - center[1])**2)
            stations_with_rebalancing.loc[idx, 'distance_to_center'] = distance_to_center
        
        # Identify outliers based on distance threshold
        quantile = stations_with_rebalancing['distance_to_center'].quantile(outlier_threshold)
        outlier_mask = stations_with_rebalancing['distance_to_center'] > quantile
        stations_to_use = stations_with_rebalancing[~outlier_mask].copy()
        outliers = stations_with_rebalancing[outlier_mask].copy()
        outliers['type'] = 'outlier'
    else:
        stations_to_use = stations_with_rebalancing.copy()
        outliers = pd.DataFrame()
    
    # Generate rebalancing routes per cluster
    for cluster_idx in range(len(cluster_centers)):
        cluster_stations = stations_to_use[stations_to_use['cluster'] == cluster_idx]
        if len(cluster_stations) <= 1:
            continue
        
        # Compute distance matrix
        n = len(cluster_stations)
        distance_matrix = np.zeros((n, n))
        station_indices = cluster_stations.index.tolist()
        
        for i, idx1 in enumerate(station_indices):
            for j, idx2 in enumerate(station_indices):
                if i != j:
                    pt1 = [cluster_stations.loc[idx1, 'longitude'], cluster_stations.loc[idx1, 'latitude']]
                    pt2 = [cluster_stations.loc[idx2, 'longitude'], cluster_stations.loc[idx2, 'latitude']]
                    distance_matrix[i, j] = distance.euclidean(pt1, pt2)
                else:
                    distance_matrix[i, j] = float('inf')
        
        # Construct route using greedy nearest neighbor approach
        if len(station_indices) > 0:
            start_node = station_indices[0]
            path = [start_node]
            current = start_node
            unvisited = set(station_indices)
            unvisited.remove(start_node)
            
            while unvisited:
                nearest = min(unvisited, key=lambda node: distance_matrix[station_indices.index(current), station_indices.index(node)])
                path.append(nearest)
                current = nearest
                unvisited.remove(nearest)
            
            path.append(start_node)  # Complete the cycle
            
            routes[cluster_idx] = {
                'path': path,
                'stations': cluster_stations
            }
    
    return routes, outliers

# Main execution function
def simulate_bike_rebalancing():
    """
    Run the full bike rebalancing simulation.
    """
    stations = generate_lyon_stations(n_stations=150)
    stations, cluster_centers = cluster_stations(stations, n_clusters=8)
    surplus_stations, deficit_stations, balanced_stations, stations_with_rebalancing = identify_rebalancing_needs(stations)
    routes_with_outliers, _ = optimize_cluster_routes(stations_with_rebalancing, cluster_centers, with_outliers=True)
    routes_without_outliers, outliers = optimize_cluster_routes(stations_with_rebalancing, cluster_centers, with_outliers=False)
    
    return stations, surplus_stations, deficit_stations, balanced_stations, routes_with_outliers, routes_without_outliers, outliers

# Run the simulation
simulate_bike_rebalancing()
