function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
vals = [ 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30 ];
numvals = size(vals,1);
errs = zeros(numvals, numvals);
for cindex = 1:numvals
    for sindex = 1:numvals
    	cval = vals(cindex);
   		sval = vals(sindex);
    	% train model X and y (training set) using cval and sval
  		model = svmTrain(X, y, cval, @(x1, x2) gaussianKernel(x1, x2, sval));
  		% compute predictions on the validation set
  		predictions = svmPredict(model, Xval);
  		% store prediction error 
  		errs(cindex, sindex) = mean(double(predictions ~= yval));
    endfor
endfor

% errs is now an array of prediction errors, with the row of the minimum being
% the index into vals of the best C and the column of the minimum being the 
% index into vals of the best sigma
[minval, row] = min(min(errs,[],2));
[minval, col] = min(min(errs,[],1));
C = vals(row);
sigma = vals(col);




% =========================================================================

end
