var trn = require('./TRNAlgorithm');
var sylvester = require('sylvester');

var M = $M([
    [1,2,3],
    [4,5,6],
    [7,8,9]
]);

trn.TRNAlgorithm(M, 1, 0, 0, 0, 0, 0, 0, 0);
