import numpy as np
import numpy.matlib 
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import random
import heapq
import sys

def random_vector(min_max_pairs):
    v = []
    for (min_val, max_val) in min_max_pairs:
        v.append(random.uniform(min_val, max_val))
    
    return np.array(v)

def trn(data_set, max_iterations, codebook_size, epsilon_i, epsilon_f, lambda_i, lambda_f, T_i, T_f):
    #connections = np.empty((codebook_size, codebook_size), dtype=object)
    connections = np.zeros((codebook_size, codebook_size), dtype=np.float16)

    dimensions = len(data_set[0])

    # generate codebook vectors
    codebook = [random_vector([(-100, 100) for k in xrange(dimensions)]) for i in xrange(codebook_size)]

    #mp_pool = Pool(processes=4)
    for t in xrange(max_iterations):
        iter_fraction = float(t) / max_iterations

        # Progress bar
        amtDone = iter_fraction * 100
        sys.stdout.write("\r%.2f%%" %amtDone)
        sys.stdout.flush()

        # Select random data point
        random_data_point = data_set[random.randint(0, len(data_set)-1)]
        
        # for each w[i] find the number of w[j] such that || x-w[j] || < || x - w[i] ||
        # array of all x - w[i]
        # array of all || x - w[i] ||
        #distances = map(lambda i: np.linalg.norm(random_data_point - codebook[i]), xrange(codebook_size))
        distances = [np.linalg.norm(random_data_point - codebook[i]) for i in xrange(codebook_size)]

        # update the codebook vectors according to the neural gas algorithm        
        lambda_val = lambda_i * ((lambda_f / lambda_i) ** iter_fraction);
        epsilon = epsilon_i * ((epsilon_f / epsilon_i) ** iter_fraction);
        for i in xrange(codebook_size):
            codebooks_near_point_i = np.sum(distances < distances[i])
            codebook[i] += epsilon * np.exp(-1 * codebooks_near_point_i / lambda_val) * (random_data_point - codebook[i])

        # find closest two codebook indices
        smallest1, smallest2 = heapq.nsmallest(2, [(v, i) for (i, v) in enumerate(distances)])
        index_smallest1, index_smallest2 = smallest1[1], smallest2[1]
        
        # create connection and refresh existing connection
        connections[index_smallest1, index_smallest2] = 1
        #connections[index_smallest2, index_smallest1] = 1

        # age all connections
        max_age = T_i * ((T_f / T_i) ** iter_fraction)

        np.putmask(connections, connections != 0, connections + 1) # Age all non-zero entries (zero => no connection)
        np.putmask(connections, connections > max_age, 0) # Remove connections greater than max age

    return codebook, connections

def connections_to_graph(connections):
    adjacency_matrix = np.zeros((len(connections), len(connections)))

    for i in xrange(len(connections)):
        for j in xrange(len(connections[i])):
            if connections[i,j] >= 1:
                adjacency_matrix[i,j] = 1
            else:
                adjacency_matrix[i,j] = 0

    G = nx.Graph(adjacency_matrix)
    return G

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

def main():
    dataset = np.genfromtxt(str(sys.argv[1]), delimiter="\t")

    num_points = len(dataset)
    import time
    t0 = time.time()
    epsilon_i = 0.3
    epsilon_f = 0.05
    lambda_i = 0.2 * num_points
    lambda_f = 0.01
    T_i = 0.1 * num_points
    T_f = 0.5 * num_points
    codebook_size = int(sys.argv[2])
    max_iter = 200 * codebook_size
    print "Constructing network on", len(dataset), "data points using", codebook_size, "codebook vectors..."

    codebook, connections = trn(dataset, max_iter, codebook_size, epsilon_i, epsilon_f, lambda_i, lambda_f, T_i, T_f)
    print ""
    print "TRN Runtime:", time.time() - t0, "seconds"

    #print "\nCodebook:"
    #for item in codebook:
    #    print item[0],",", item[1],",",item[2]

    print "\nConnections:"

    G = connections_to_graph(connections)
    
    print "Drawing graph..."
    print G.edges()
    draw_graph(G, "graph.png")
    output_json(G)
    print "Done!"
    #output_json(codebook, connections)

if __name__ == "__main__":
    main()

