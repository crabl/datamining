import numpy as np
import numpy.matlib 
#from numba import autojit
import random
import heapq

def random_vector(min_max_pairs):
    v = []
    for (min_val, max_val) in min_max_pairs:
        v.append(random.uniform(min_val, max_val))
    
    return np.array(v)

# Returns [w, C, tc]
#def trn(X, t_max, N, epsilon_i, epsilon_f, Ti, Tf, lambda_i, lambda_f):
#@autojit
def trn(data_set, max_iterations, codebook_size, epsilon, lambda_val, max_age):
    connections = np.empty((codebook_size, codebook_size), dtype=object)
    # generate codebook vectors
    codebook = []
    for i in range(0, codebook_size):
        codebook.append(
            random_vector([
                (0, 10),
                (0, 10),
                (0, 10)]))

    for __data_point__ in data_set:
        random_data_point = data_set[random.randint(0, len(data_set) - 1)] # choose a random data point

        for t in range(0, max_iterations):
            codebooks_near_point = np.zeros(codebook_size)
            
            # for each w[i] find the number of w[j] such that || x-w[j] || < || x - w[i] ||
            # array of all x - w[i]
            # array of all || x - w[i] ||
            """
            for i in range(0, codebook_size):
            for j in range(0, codebook_size):
            if j == i:
            pass
            else:
            if np.linalg.norm(random_data_point - codebook[j]) < np.linalg.norm(random_data_point - codebook[i]):
            codebooks_near_point[i] += 1
            """
            distances = []

            for i in range(0, codebook_size):
                distances.append(np.linalg.norm(random_data_point - codebook[i]))
        
            for i in range(0, codebook_size):
                codebooks_near_point[i] = np.sum(distances < distances[i])
                # update the codebook vectors according to the neural gas algorithm
                #for i in range(0, codebook_size):
                codebook[i] = codebook[i] + epsilon * np.exp(-1 * codebooks_near_point[i] / lambda_val) * (random_data_point - codebook[i])

            # find closest two codebook indices <- SOMETHING WRONG HERE
            smallest1, smallest2 = heapq.nsmallest(2, [(v, i) for (i, v) in enumerate(distances)])
            index_smallest1, index_smallest2 = smallest1[1], smallest2[1]
            connections[index_smallest1, index_smallest2] = [1, 0]
            connections[index_smallest2, index_smallest1] = [1, 0]
            
            # age all connections
            for row in range(0, len(connections)):
                for col in range(0, len(connections[row])):
                    if connections[row][col] != None:
                        connections[row][col][1] += 1
                        if connections[row][col][1] > max_age:
                            connections[row][col] = None


def main():
    num_points = 200
    dataset = []
    for i in range(0, num_points):
        dataset.append(random_vector([
            (1,10),
            (1,10),
            (1,10)]))

    import time
    t0 = time.time()
    trn(dataset, int(0.2*num_points), 20, 0.2, 0.2, 10)
    print time.time() - t0

if __name__ == "__main__":
    main()


"""
    for i=1:tmax:
        
        %step2
        v=X(:,floor(N*rand)+1);
        
        %step3
        dist=[];
        dist=sum((w-repmat(v,1,size(w,2))).^2,1)';
        [dist2,index]=sort(dist);
        [index2,kindex]=sort(index);
        
        %step4
        lambda=lambdai*((lambdaf/lambdai)^(i/tmax));
        epsz=epszi*((epszf/epszi)^(i/tmax));
        for k=1:n:
            w(:,k)=w(:,k)+epsz.*exp(-(kindex(k)-1)/lambda)*(v-w(:,k));

    
    %step5
    C(index(1),index(2))=1;
    C(index(2),index(1))=1;
    t(index(1),index(2))=0;
    t(index(2),index(1))=0;
    tc(index(1),index(2))=tc(index(1),index(2))+(i/tmax);%%%%%%%%%%%%%%%%%%
    tc(index(2),index(1))=tc(index(2),index(1))+(i/tmax);%%%%%%%%%%%%%%%%%%
    
    
    %step6
    connectsh=find(C(index(1),:));
    t(index(1),connectsh)=t(index(1),connectsh)+1;
    connectsv=find(C(:,index(1)));
    t(connectsv,index(1))=t(connectsv,index(1))+1;
    
    
   %step7
   T=Ti*(Tf/Ti)^(i/tmax);
   deleteh=find(t(index(1),:)>T);
   C(index(1),deleteh)=0;
   t(index(1),deleteh)=0;
   deletev=find(t(:,index(1))>T);
   C(deletev,index(1))=0;
   t(deletev,index(1))=0;
  
   
end

figdrawn(X,w,C,'TRN')

"""
