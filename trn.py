#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlb
import networkx as nx
from networkx.readwrite import json_graph, write_gexf
#from mayavi import mlab
import random
import heapq
import sys
import sklearn.preprocessing as skp
from scipy.linalg import eigh as largest_eigh
import scipy.spatial.distance as dist

def random_vector(min_max_pairs):
    v = []
    for (min_val, max_val) in min_max_pairs:
        v.append(random.uniform(min_val, max_val))

    return np.array(v)

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

def TRN(data_set, max_iterations, codebook_size, epsilon_i, epsilon_f, lambda_i, lambda_f, T_i, T_f):
    connections = np.zeros((codebook_size, codebook_size), dtype=np.uint16)
    dimensions = len(data_set[0])

    # generate codebook vectors
    codebook = np.array([random_vector([(-100, 100) for k in xrange(dimensions)]) for i in xrange(codebook_size)])

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
        for i in xrange(codebook_size):
            codebooks_near_point_i = (distances < distances[i]).sum()
            codebook[i] += epsilon * np.exp(-1 * codebooks_near_point_i / lambda_val) * (random_data_point - codebook[i])

        # find closest two codebook indices
        smallest1, smallest2 = heapq.nsmallest(2, [(v, i) for (i, v) in enumerate(distances)])
        index_smallest1, index_smallest2 = smallest1[1], smallest2[1]

        # create connection and refresh existing connection
        connections[index_smallest1, index_smallest2] = 1

        # age all connections
        max_age = T_i * ((T_f / T_i) ** iter_fraction)
        np.putmask(connections, connections != 0, connections + 1) # Age all non-zero entries (zero => no connection)
        np.putmask(connections, connections > max_age, 0) # Remove connections greater than max age

    return codebook, connections

def connections_to_graph(connections, codebook):
    np.putmask(connections, connections > 1, 1)
    G = nx.Graph(connections)
    x_dict = dict(zip(range(len(codebook[:,0])), codebook[:,0]))
    y_dict = dict(zip(range(len(codebook[:,1])), codebook[:,1]))
    nx.set_node_attributes(G, 'x', x_dict)
    nx.set_node_attributes(G, 'y', y_dict)
    return G

#def draw_mayavi_graph(G, file_name):


def draw_graph(G, file_name):
    nx.draw(G)
    plt.savefig(file_name)

def output_json(G):
    import json
    f = open("graph.json", "w")
    data = json_graph.node_link_data(G)
    serialized_data = json.dumps(data)
    f.write(serialized_data)
    f.close()

def output_gexf(G):
    write_gexf(G, "graph.gexf")

def main(fileName, codebookSize):
    raw_dataset = np.genfromtxt(str(fileName), delimiter="\t")
    dataset = skp.normalize(raw_dataset) # only needed for TRNMAP
    #dataset = raw_dataset
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

    codebook, connections = TRN(dataset, max_iter, codebook_size, epsilon_i, epsilon_f, lambda_i, lambda_f, T_i, T_f)
    print ""
    print "TRN Runtime:", time.time() - t0, "seconds"

    print "Scaling down to 2 dimensions..."
    scaled_codebook = MDS(codebook, 2) # scale to 2-dimensional space


    print "Drawing graph..."
    M = connections_to_graph(connections, scaled_codebook)
    print "Number of subgraphs:", nx.number_connected_components(M)
    print M.nodes()
    print M.edges()
    draw_graph(M, "graph.png")
    output_json(M)
    #output_gexf(M)
    print "Done!"

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])










