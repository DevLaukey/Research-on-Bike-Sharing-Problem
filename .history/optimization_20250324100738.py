import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import networkx as nx
import random
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate simulated bike stations in Lyon
def generate_lyon_stations(n_stations=150):
    # Approximate coordinates for Lyon city center and surrounding areas
    # Format: [longitude, latitude]
    lyon_center = [4.835, 45.76]  # Lyon city center coordinates
    
    # Generate stations with a concentration around city center and some spread
    longs = np.random.normal(lyon_center[0], 0.03, n_stations)
    lats = np.random.normal(lyon_center[1], 0.02, n_stations)
    
    # Generate random inventory values (current number of bikes)
    current_inventory = np.random.randint(0, 20, n_stations)
    
    # Generate target inventory values (optimal number of bikes)
    target_inventory = np.random.randint(5, 20, n_stations)
    
    # Calculate imbalance (positive means surplus, negative means deficit)
    imbalance = current_inventory - target_inventory
    
    # Create DataFrame
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
    # Extract coordinates for clustering
    coordinates = stations[['longitude', 'latitude']].values
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    stations['cluster'] = kmeans.fit_predict(coordinates)
    
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    return stations, cluster_centers

# Function to identify stations that need rebalancing
def identify_rebalancing_needs(stations, threshold=3):
    # Stations with significant surplus (need pick-up)
    surplus_stations = stations[stations['imbalance'] > threshold].copy()
    surplus_stations['type'] = 'surplus'
    
    # Stations with significant deficit (need drop-off)
    deficit_stations = stations[stations['imbalance'] < -threshold].copy()
    deficit_stations['type'] = 'deficit'
    
    # Balanced stations
    balanced_stations = stations[(stations['imbalance'] >= -threshold) & 
                               (stations['imbalance'] <= threshold)].copy()
    balanced_stations['type'] = 'balanced'
    
    # Identify outlier stations (furthest from cluster centers)
    stations_with_rebalancing = pd.concat([surplus_stations, deficit_stations])
    
    return surplus_stations, deficit_stations, balanced_stations, stations_with_rebalancing

# Function to optimize rebalancing routes within clusters
def optimize_cluster_routes(stations_with_rebalancing, cluster_centers, with_outliers=True):
    routes = {}
    
    # Identify potential outliers (5% of stations furthest from their cluster centers)
    if not with_outliers:
        outlier_threshold = 0.95  # Top 5% are outliers
        
        # Calculate distances to cluster centers
        for idx, station in stations_with_rebalancing.iterrows():
            cluster_idx = station['cluster']
            center = cluster_centers[cluster_idx]
            distance_to_center = np.sqrt((station['longitude'] - center[0])**2 + 
                                         (station['latitude'] - center[1])**2)
            stations_with_rebalancing.loc[idx, 'distance_to_center'] = distance_to_center
        
        # Mark outliers
        quantile = stations_with_rebalancing['distance_to_center'].quantile(outlier_threshold)
        outlier_mask = stations_with_rebalancing['distance_to_center'] > quantile
        stations_to_use = stations_with_rebalancing[~outlier_mask].copy()
        outliers = stations_with_rebalancing[outlier_mask].copy()
        outliers['type'] = 'outlier'
    else:
        stations_to_use = stations_with_rebalancing.copy()
        outliers = pd.DataFrame()
    
    # For each cluster, find optimal route
    for cluster_idx in range(len(cluster_centers)):
        cluster_stations = stations_to_use[stations_to_use['cluster'] == cluster_idx]
        
        if len(cluster_stations) <= 1:
            continue
            
        # Create distance matrix between all stations in cluster
        n = len(cluster_stations)
        distance_matrix = np.zeros((n, n))
        station_indices = cluster_stations.index.tolist()
        
        for i, idx1 in enumerate(station_indices):
            for j, idx2 in enumerate(station_indices):
                if i != j:
                    pt1 = [cluster_stations.loc[idx1, 'longitude'], cluster_stations.loc[idx1, 'latitude']]
                    pt2 = [cluster_stations.loc[idx2, 'longitude'], cluster_stations.loc[idx2, 'latitude']]
                    dist = distance.euclidean(pt1, pt2)
                    distance_matrix[i, j] = dist
                else:
                    distance_matrix[i, j] = float('inf')  # Don't connect to self
        
        # Create a graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, idx in enumerate(station_indices):
            G.add_node(idx)
        
        # Add edges with distance weights
        for i, idx1 in enumerate(station_indices):
            for j, idx2 in enumerate(station_indices):
                if i != j:
                    G.add_edge(idx1, idx2, weight=distance_matrix[i, j])
        
        # Find approximate solution to TSP with greedy algorithm
        if len(G.nodes) > 0:
            start_node = station_indices[0]
            path = [start_node]
            current = start_node
            unvisited = set(station_indices)
            unvisited.remove(start_node)
            
            while unvisited:
                # Find the nearest unvisited node
                nearest = min(unvisited, key=lambda node: G[current][node]['weight'])
                path.append(nearest)
                current = nearest
                unvisited.remove(nearest)
            
            # Complete the circuit by going back to start
            path.append(start_node)
            
            # Store the optimized route
            routes[cluster_idx] = {
                'path': path,
                'stations': cluster_stations
            }
    
    return routes, outliers

# Function to create the visualization
def visualize_rebalancing(stations, surplus_stations, deficit_stations, balanced_stations, 
                         routes_with_outliers, routes_without_outliers, outliers):
    plt.figure(figsize=(20, 8))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # Define colors for clusters
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Create first subplot: Station Target Distribution
    ax1 = plt.subplot(gs[0])
    
    # Normalize imbalance for coloring
    norm = plt.Normalize(vmin=-10, vmax=10)
    cmap = plt.cm.RdBu_r  # Red-Blue colormap (red for deficit, blue for surplus)
    
    # Plot all stations with size proportional to target inventory and color by imbalance
    for idx, station in stations.iterrows():
        size = 20 + station['target_inventory'] * 2  # Scale size based on target
        color = cmap(norm(station['imbalance']))
        ax1.scatter(station['longitude'], station['latitude'], s=size, c=[color], alpha=0.7, 
                   edgecolors='k', linewidths=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, label='Imbalance (Surplus/Deficit)')
    cbar.set_ticks([-10, -5, 0, 5, 10])
    cbar.set_ticklabels(['-10 (Deficit)', '-5', '0', '5', '10 (Surplus)'])
    
    # Create legend for station sizes
    size_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='5 bikes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='10 bikes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='15 bikes')
    ]
    ax1.legend(handles=size_legend_elements, title="Target Inventory", loc='upper right')
    
    ax1.set_title('(a) Station Target Distribution')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Create second subplot: Route with outliers
    ax2 = plt.subplot(gs[1])
    
    # Plot routes with outliers
    for cluster_idx, route_info in routes_with_outliers.items():
        color = colors[cluster_idx % len(colors)]
        path = route_info['path']
        cluster_stations = route_info['stations']
        
        # Plot route
        for i in range(len(path) - 1):
            start_idx = path[i]
            end_idx = path[i + 1]
            
            x1, y1 = cluster_stations.loc[start_idx, 'longitude'], cluster_stations.loc[start_idx, 'latitude']
            x2, y2 = cluster_stations.loc[end_idx, 'longitude'], cluster_stations.loc[end_idx, 'latitude']
            
            ax2.plot([x1, x2], [y1, y2], c=color, linewidth=1.5, alpha=0.8)
        
        # Plot stations
        for idx, station in cluster_stations.iterrows():
            if station['type'] == 'surplus':
                marker = 'o'  # Circle for stations that need pick-up
            else:
                marker = 'o'  # Also circle for stations that need drop-off, but we'll differentiate with color
                
            ax2.scatter(station['longitude'], station['latitude'], c=color, s=60, 
                       marker=marker, edgecolors='k', linewidths=0.5, alpha=0.8)
    
    # Plot balanced stations
    for idx, station in balanced_stations.iterrows():
        ax2.scatter(station['longitude'], station['latitude'], c='black', s=30, 
                   marker='d', edgecolors='k', linewidths=0.5, alpha=0.4)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=10, label='Origin Station'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=10, label='Optimized Station'),
        Line2D([0], [0], marker='x', color='black', markersize=10, label='Outlier Station'),
        Line2D([0], [0], marker='d', color='black', markersize=8, label='Balanced Station')
    ]
    ax2.legend(handles=legend_elements, loc='lower left')
    
    ax2.set_title(f'(b) Route with outliers (VN = 8)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    # Create third subplot: Route without outliers
    ax3 = plt.subplot(gs[2])
    
    # Plot routes without outliers
    for cluster_idx, route_info in routes_without_outliers.items():
        color = colors[cluster_idx % len(colors)]
        path = route_info['path']
        cluster_stations = route_info['stations']
        
        # Plot route
        for i in range(len(path) - 1):
            start_idx = path[i]
            end_idx = path[i + 1]
            
            x1, y1 = cluster_stations.loc[start_idx, 'longitude'], cluster_stations.loc[start_idx, 'latitude']
            x2, y2 = cluster_stations.loc[end_idx, 'longitude'], cluster_stations.loc[end_idx, 'latitude']
            
            ax3.plot([x1, x2], [y1, y2], c=color, linewidth=1.5, alpha=0.8)
        
        # Plot stations
        for idx, station in cluster_stations.iterrows():
            if station['type'] == 'surplus':
                marker = 'o'  # Circle for stations that need pick-up
            else:
                marker = 'o'  # Also circle for stations that need drop-off
                
            ax3.scatter(station['longitude'], station['latitude'], c=color, s=60, 
                       marker=marker, edgecolors='k', linewidths=0.5, alpha=0.8)
    
    # Plot outliers
    for idx, station in outliers.iterrows():
        ax3.scatter(station['longitude'], station['latitude'], c='black', s=50, 
                   marker='x', alpha=0.8)
    
    # Plot balanced stations
    for idx, station in balanced_stations.iterrows():
        ax3.scatter(station['longitude'], station['latitude'], c='black', s=30, 
                   marker='d', edgecolors='k', linewidths=0.5, alpha=0.4)
    
    # Create legend
    ax3.legend(handles=legend_elements, loc='lower left')
    
    ax3.set_title(f'(c) Route without outliers (VN = 12)')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    
    plt.suptitle('Figure 9: Vehicle Rebalancing Route Optimization (Case Study)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return plt

# Main execution function
def simulate_bike_rebalancing():
    # 1. Generate simulated stations
    stations = generate_lyon_stations(n_stations=150)
    
    # 2. Cluster stations
    stations, cluster_centers = cluster_stations(stations, n_clusters=8)
    
    # 3. Identify rebalancing needs
    surplus_stations, deficit_stations, balanced_stations, stations_with_rebalancing = (
        identify_rebalancing_needs(stations)
    )
    
    # 4. Optimize routes with outliers
    routes_with_outliers, _ = optimize_cluster_routes(
        stations_with_rebalancing, cluster_centers, with_outliers=True
    )
    
    # 5. Optimize routes without outliers
    routes_without_outliers, outliers = optimize_cluster_routes(
        stations_with_rebalancing, cluster_centers, with_outliers=False
    )
    
    # 6. Visualize results
    plt_figure = visualize_rebalancing(
        stations, surplus_stations, deficit_stations, balanced_stations,
        routes_with_outliers, routes_without_outliers, outliers
    )
    
    return plt_figure

# Run the simulation
fig = simulate_bike_rebalancing()
plt.savefig('bike_rebalancing_simulation.png', dpi=300, bbox_inches='tight')
plt.show()

# To modify the simulation for more realistic Lyon data:
# 1. Replace generate_lyon_stations() with actual station data
# 2. Adjust the threshold in identify_rebalancing_needs() based on real capacity needs
# 3. Tune the number of clusters based on actual operations