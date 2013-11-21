#!/usr/bin/env python

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

def random_vector(min_max_pairs):
    v = []
    for (min_val, max_val) in min_max_pairs:
        v.append(random.uniform(min_val, max_val))

    return np.array(v)

def geodesic_distance(codebook, connections):
    distances = dist.squareform(dist.pdist(codebook, 'euclidean'))
    sparse_distances = sparse.csr_matrix(distances * connections)
    geo = sparse.csgraph.dijkstra(sparse_distances,indices=range(0, len(codebook)))
    return geo

def connect_graph(codebook, connections):
    raise NotImplementedError



def MDS(codebook, dimensions):
    num_points = len(codebook)
    P = dist.squareform(dist.pdist(codebook, 'euclidean'))**2 # matrix of SQUARED distances (really this is P^2)
    I = np.identity(num_points)
    ONE = np.ones((num_points, num_points))
    J = I - (1./num_points) * ONE
    B = -0.5 * J * P * J
    # Calculate the d largest e-vals (L) and corresponding e-vects (E)
    (L, E) = largest_eigh(B, eigvals=(num_points - dimensions, num_points - 1))
    L = np.sqrt(np.flipud(L)) # Sort from highest to lowest and square root
    E = np.fliplr(E) # sort from largest to smallest positive e-vectors
    return -1*(E*L) # don't know why we need the -1...

def TRN(data_set, max_iterations, codebook_size, epsilon_i, epsilon_f, lambda_i, lambda_f, T_i, T_f, folder_path):
    connections = np.zeros((codebook_size, codebook_size), dtype=np.uint16)
    dimensions = len(data_set[0])

    #3d view settings
    elevation = 30
    azimuth = 45

    # generate codebook vectors
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

        # for each w[i] find the number of w[j] such that || x-w[j] || < || x - w[i] ||
        # array of all x - w[i]
        # array of all || x - w[i] ||
        # using numpy vectorization
        V = random_data_point - codebook
        distances = np.sqrt(np.multiply(V, V)).sum(axis=1)

        # update the codebook vectors according to the neural gas algorithm
        lambda_val = lambda_i * ((lambda_f / lambda_i) ** iter_fraction);
        epsilon = epsilon_i * ((epsilon_f / epsilon_i) ** iter_fraction);

        # This is a huge bottleneck... make it faster
        codebooks_near = np.array([np.exp(-1 * ((distances<distances[i]).sum())/lambda_val) for i in xrange(codebook_size)])
        coefficients = epsilon * codebooks_near
        codebook += coefficients.reshape((len(coefficients),1)) * V

        #for i in xrange(codebook_size):
        #    codebooks_near_point_i = (distances < distances[i]).sum()
        #    codebook[i] += epsilon * np.exp(-1 * codebooks_near_point_i / lambda_val) * (random_data_point - codebook[i])

        # find closest two codebook indices
        smallest1, smallest2 = heapq.nsmallest(2, [(v, i) for (i, v) in enumerate(distances)])
        index_smallest1, index_smallest2 = smallest1[1], smallest2[1]

        # create connection and refresh existing connection
        connections[index_smallest1, index_smallest2] = 1

        # age all connections
        max_age = T_i * ((T_f / T_i) ** iter_fraction)
        np.putmask(connections, connections != 0, connections + 1) # Age all non-zero entries (zero => no connection)
        np.putmask(connections, connections > max_age, 0) # Remove connections greater than max age

        # Intermediary graph
        if t % 50 == 0:
            G_i = nx.Graph()
            G_i, pos = connections_to_graph(connections, codebook)
            azimuth += 0.7
            azimuth = azimuth % 360
            draw_graph(G_i, pos, data_set, folder_path+"/graph_"+str(t)+".jpg", elevation, azimuth)

    return codebook, connections

def connections_to_graph(connections, codebook):
    np.putmask(connections, connections > 1, 1)
    G = nx.Graph(connections)
    positions = dict(zip(range(0,len(codebook)), codebook))
    return G, positions

#def draw_mayavi_graph(G, file_name):


def draw_graph(G, positions, dataset, file_name, el, az):
    plt.clf() # Clear the figure
    fig = plt.figure()
    plt.figure.max_num_figures = 20
    ax = fig.add_subplot(111, projection='3d')

    #draw lines between codebook vectors
    lines = [[positions[x],positions[y]] for (x,y) in G.edges()]
    for line in lines:
        x,y,z = zip(*line)
        ax.plot(x,y,z, color='#000000', ls='-', alpha=0.25)

    ax.plot(*zip(*[positions[i] for i in positions.keys()]), marker='o', color='#BEF202', ls='')
    ax.plot(*zip(*dataset), marker='.', color='b', ls='')

    ax.view_init(el, az)
    #plt.autoscale(True, "both", True)
    #plt.plot(*zip(*dataset), marker='.', color='b', ls='')
    #nx.draw(G, pos=positions, node_color="#BEF202", dim=3)
    #fig.savefig(file_name, filetype="jpg")

def output_json(G, file_name):
    import json
    f = open(file_name, "w")
    data = json_graph.node_link_data(G)
    serialized_data = json.dumps(data)
    f.write(serialized_data)
    f.close()

def output_gexf(G):
    write_gexf(G, "graph.gexf")

def main(fileName, codebookSize):
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

    folder_path = sys.argv[3]

    codebook, connections = TRN(dataset, max_iter, codebook_size, epsilon_i, epsilon_f, lambda_i, lambda_f, T_i, T_f, folder_path)
    np.putmask(connections, connections > 1, 1.) # Needs to be a binary matrix
    print ""
    print geodesic_distance(codebook, connections)
    scio.savemat("codebook.mat", mdict={'codebook':codebook})
    print ""
    scio.savemat("connections.mat", mdict={'connections': connections})
    print ""
    print "TRN Runtime:", time.time() - t0, "seconds"

    print "Scaling down to 2 dimensions..."
    scaled_codebook = MDS(codebook, 2) # scale to 2-dimensional space


    print "Drawing graph..."
    M = connections_to_graph(connections, scaled_codebook)
    #print "Number of subgraphs:", nx.number_connected_components(M)
    #print M.nodes()
    #print M.edges()
    #draw_graph(M, "graph.png")
    #output_json(M, "graph.json")
    #output_gexf(M)
    print "Done!"

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])










