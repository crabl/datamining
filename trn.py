#!/usr/bin/env python

# TRNMap Algorithm by Vass-Fogarassy, Abonyi, et al. [2008]
# Implemented in Numpy by Christopher Rabl

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph, write_gexf
import random
import heapq
import sys
import sklearn.preprocessing as skp
from scipy.linalg import eigh as largest_eigh
import scipy.spatial.distance as dist
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio
import scipy.sparse as sparse
import ubigraph

def random_vector(min_max_pairs):
    v = []
    for (min_val, max_val) in min_max_pairs:
        v.append(random.uniform(min_val, max_val))

    return np.array(v)

def geodesic_distance(codebook, connections):
    distances = dist.squareform(dist.pdist(codebook, 'euclidean'))
    graph_edges = np.multiply(distances, connections)
    geo = sparse.csgraph.dijkstra(graph_edges,indices=range(0, len(codebook)))
    return geo

def connect_graph(codebook, connections):
    dijkstra_distances = geodesic_distance(codebook, connections)
    np.putmask(dijkstra_distances, dijkstra_distances==np.inf, 0) # replace all infinite edges with 0

    distances = dist.squareform(dist.pdist(codebook, 'euclidean'))

    candidate_edges = distances - dijkstra_distances
    np.putmask(candidate_edges, candidate_edges <= 0, np.nan)

    num_subgraphs = len(np.unique(candidate_edges.argmin(0)))
    print "    Remaining subgraphs: ",

    # While we have unconnected subgraphs
    while num_subgraphs > 1:
        print num_subgraphs, 

        # Find the smallest entry
        shortest_new_edge = np.nanmin(candidate_edges) 
        # Find the indices of that entry
        new_edge_indices = np.where(candidate_edges==shortest_new_edge) 

        if len(new_edge_indices[0]) > 1:
            new_edge_indices = new_edge_indices[0]

        # Add the new edge to the connection graph
        connections[new_edge_indices[0], new_edge_indices[1]] = 1 
        connections[new_edge_indices[1], new_edge_indices[0]] = 1

        dijkstra_distances = geodesic_distance(codebook, connections)
        # Replace all infinite edges from Dijkstra's output with 0 mask
        np.putmask(dijkstra_distances, dijkstra_distances==np.inf, 0)

        # Recompute candidate edge set now that we've hopefully added a connection
        distances = dist.squareform(dist.pdist(codebook, 'euclidean'))
        candidate_edges = distances - dijkstra_distances
        np.putmask(candidate_edges, candidate_edges <= 0, np.nan)
        num_subgraphs = len(np.unique(candidate_edges.argmin(0)))

    print "Done!",

    return connections

def MDS(codebook, dimensions):
    num_points = len(codebook)

    # matrix of SQUARED distances (really this is P^2)
    P = dist.squareform(dist.pdist(codebook, 'euclidean'))**2 
    I = np.identity(num_points)
    ONE = np.ones((num_points, num_points))
    J = I - (1./num_points) * ONE # Centering matrix for MDS
    B = -0.5 * J * P * J

    # Calculate the d largest eigenvalues (L) and corresponding eigenvectors (E)
    (L, E) = largest_eigh(B, eigvals=(num_points - dimensions, num_points - 1))

    # Sort from highest to lowest and square root
    L = np.sqrt(np.flipud(L)) 

    # Sort from largest to smallest positive eigenvectors
    E = np.fliplr(E) 

    return -1*(E*L) # don't know why we need the -1...

def TRN(data_set, max_iterations, codebook_size, epsilon_i, epsilon_f, lambda_i, lambda_f, T_i, T_f):
    connections = np.zeros((codebook_size, codebook_size), dtype=np.uint16)
    dimensions = len(data_set[0])

    # Generate codebook vectors
    codebook = np.array([random_vector([(0, 1) for k in xrange(dimensions)]) for i in xrange(codebook_size)])

    prevDone = 0
    for t in xrange(max_iterations):
        iter_fraction = float(t) / max_iterations

        # Progress bar
        amtDone = iter_fraction * 100
        sys.stdout.write("\r%.1f%%" %amtDone)
        sys.stdout.flush()

        # Select random data point
        random_data_point = data_set[random.randint(0, len(data_set)-1)]

        # For each w[i] find the number of w[j] such that || x-w[j] || < || x - w[i] ||
        # array of all x - w[i]
        # array of all || x - w[i] ||
        # using numpy vectorization
        V = random_data_point - codebook
        distances = np.sqrt(np.multiply(V, V)).sum(axis=1)

        # Update the codebook vectors according to the neural gas algorithm
        lambda_val = lambda_i * ((lambda_f / lambda_i) ** iter_fraction);
        epsilon = epsilon_i * ((epsilon_f / epsilon_i) ** iter_fraction);

        # This is a huge bottleneck... make it faster
        codebooks_near = np.array([np.exp(-1 * ((distances<distances[i]).sum())/lambda_val) for i in xrange(codebook_size)])
        coefficients = epsilon * codebooks_near
        codebook += coefficients.reshape((len(coefficients),1)) * V

        # Find closest two codebook indices
        smallest1, smallest2 = heapq.nsmallest(2, [(v, i) for (i, v) in enumerate(distances)])
        index_smallest1, index_smallest2 = smallest1[1], smallest2[1]

        # Create connection and refresh existing connection
        connections[index_smallest1, index_smallest2] = 1

        # Age all connections
        max_age = T_i * ((T_f / T_i) ** iter_fraction)
        np.putmask(connections, connections != 0, connections + 1) # Age all non-zero entries (zero => no connection)
        np.putmask(connections, connections > max_age, 0) # Remove connections greater than max age

    return codebook, connections

def connections_to_graph(codebook_2d, connections):
    np.putmask(connections, connections > 1, 1)
    G = nx.Graph(connections)
    x_dict = dict(zip(range(len(codebook_2d[:,0])), codebook_2d[:,0]))
    y_dict = dict(zip(range(len(codebook_2d[:,1])), codebook_2d[:,1]))
    nx.set_node_attributes(G, 'x', x_dict)
    nx.set_node_attributes(G, 'y', y_dict)

    positions = dict(zip(range(0,len(codebook_2d)), codebook_2d))

    return G, positions

def draw_graph_3d(G, positions, dataset, file_name, el, az):
    plt.clf() # Clear the figure
    fig = plt.figure()
    plt.figure.max_num_figures = 20
    ax = fig.add_subplot(111, projection='3d')

    # Draw lines between codebook vectors
    lines = [[positions[x],positions[y]] for (x,y) in G.edges()]
    for line in lines:
        x,y,z = zip(*line)
        ax.plot(x,y,z, color='#000000', ls='-', alpha=0.25)

    ax.plot(*zip(*[positions[i] for i in positions.keys()]), marker='o', color='#BEF202', ls='')
    ax.plot(*zip(*dataset), marker=',', color='b', ls='')

    ax.view_init(el, az)
    fig.savefig(file_name, filetype="jpg")

def output_json(G, file_name):
    import json
    f = open(file_name, "w")
    data = json_graph.node_link_data(G)
    serialized_data = json.dumps(data)
    f.write(serialized_data)
    f.close()

def output_gexf(G):
    write_gexf(G, "graph.gexf")

def main(fileName, codebookSize, scaleToDimension, drawUbigraph):
    raw_dataset = np.genfromtxt(str(fileName), delimiter="\t")
    #dataset = skp.normalize(raw_dataset) # only needed for TRNMAP
    dataset = raw_dataset
    num_points = len(dataset)
    import time
    t0 = time.time()
    epsilon_i = 0.3
    epsilon_f = 0.05
    lambda_i = 0.2 * num_points
    lambda_f = 0.01
    T_i = 0.1 * num_points
    T_f = 0.5 * num_points
    codebook_size = int(codebookSize)
    max_iter = 200 * codebook_size
    print "Constructing network on", len(dataset), "data points using", codebook_size, "codebook vectors..."

    # Calculate TRN
    codebook, connections = TRN(dataset, max_iter, codebook_size, epsilon_i, epsilon_f, lambda_i, lambda_f, T_i, T_f)

    # Change connections to a boolean array, since it currently stores connection ages
    np.putmask(connections, connections > 1, 1.) # Needs to be a binary matrix
    print ""

    # Connect the graph
    print "Connecting unconnected edges..."
    connections = connect_graph(codebook, connections)
    print ""
    print "Scaling down to", scaleToDimension,"dimensions..."

    # Scale codebook down to two dimensions using MDS
    scaled_codebook = MDS(codebook, scaleToDimension)
    print "TRNMap Runtime:", time.time() - t0, "seconds"

    print "Exporting to JSON..."
    G, scaled_positions = connections_to_graph(scaled_codebook, connections)
    #nx.draw(N, pos=scaled_positions, node_color="#BEF202", dim=2)
    #plt.savefig("graph_2d.jpg", filetype="jpg")

    # Output graph to JSON that Sigma/D3.js can read
    output_json(G, "graph.json")

    # Draw Ubigraph if user requests it
    if drawUbigraph:
        print "Drawing in UbiGraph..."
        vertices = {}
        edges = []
        U = ubigraph.Ubigraph()
        U.clear()

        # Plot vertices
        for vertex in G.nodes():
            vertices[vertex] = U.newVertex(vertex)

        # Plot edges
        for edge in G.edges():
            edges.append(U.newEdge(vertices[edge[0]], vertices[edge[1]]))

    print "Done!"

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), bool(sys.argv[4]))

