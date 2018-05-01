function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
[m n] = size(X);

% create an m x n x K matrix for the samples, with K copies of X as the 3rd dimension
repX = repmat(X,1,1,K);

% create an m x n x K matrix for the centroids. Each centroid is replicated m times 
% and that is made a matrix in the first 2 dimenions. There are K such matrixes, which
% is the third dimension
repK = reshape(repelems(centroids', [ [1:(K*n)]; repmat(m,K*n,1)'])',m,n,K);

% create an m x K matrix which is the euclidian distances of each X from each centroid
MK = reshape(sqrt(sum((repX - repK) .^ 2,2)),m, K);

% given an m x K matrix (MK) which contains the euclidian distances for each 
% of the m samples in X across the K centroids, get the indexes of minimum 
% values from each row, This will give the index number of the closest centroid.
[_ idx] = min(MK, [], 2);

% =============================================================

end