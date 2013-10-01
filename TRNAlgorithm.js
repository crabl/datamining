var sylvester = require('sylvester');

var RepMat = function(A, m, n) {
    var rows = A.rows();
    var cols = A.cols();

    var matrix = new Array(rows * m);
    for(var i = 0; i < rows * m; i++) {
	var column = new Array(cols * n);
	for(var j = 0; j < cols * n; j++) {
	    column[j] = A.e(i%rows+1,j%rows+1);
	}

	matrix[i] = column;
    }

    return $M(matrix);
};


exports.TRNAlgorithm = function(X, t_max, n, epsilon_i, epsilon_f, T_i, T_f, lambda_i, lambda_f) {
    var W = [];
    var C = [];
    
    var dim = X.rows();
    var N = X.cols(); 
    var range = 1.2; // Factor of the data range

    for(var i = 0; i < t_max; i++) {
	v = X.col(Math.floor(Math.random() * N) + 1); // Select a random input vector (returned as a row vector)
	squared = X.map(function(x) { return x*x; });

    }
    

    return [W, C];
};
