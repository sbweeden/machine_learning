function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%set up y_matrix to be a vector of outputs for each y result (5000 x 10)
y_matrix = eye(num_labels)(y,:);

% Add ones to the X data matrix to get a1 (5000 x 401)
a1 = [ones(m, 1) X];

% z2 is the product of a1 and Theta1' (5000x25)
z2 = a1 * Theta1';

% a2 is then 1's prepended to the g(z2) (5000x26)
a2 = [ones(m,1) sigmoid(z2)];

% z3 equals the product of a2 and Theta2' (5000x10)
z3 = a2 * Theta2';

% a3 is the result of passing z3 through g() (5000x10)
a3 = sigmoid(z3);

stuff = (-1 * y_matrix) .* log(a3) - ((1 - y_matrix) .* log(1 - a3));
J = sum(stuff(:)) / m;


% regularization - ignore first column of both Theta1 and Theta2 so that we don't regularize theta0
rTheta1=Theta1(:,2:end); % 25x400
rTheta2=Theta2(:,2:end); % 10x25
reg = (lambda / (2*m)) * (sum((rTheta1 .^ 2)(:)) + sum((rTheta2 .^ 2)(:)));
J = J + reg;

% d3 is difference between a3 and the y_matrix (5000x10)
d3 = a3 - y_matrix;

% d2 = product of d3 and Theta2 (ignoring bias units) then multiplied element-wise by the g′() of z2
d2 = (d3 * rTheta2) .* sigmoidGradient(z2); % 5000x25

% Delta2 is the product of d3 and a2
Delta2 = d3' * a2; % 10x26
regD2 = (lambda / m) * Theta2; %10x26
regD2 = [zeros(rows(regD2),1) regD2(:,2:end)];

% Delta1 is the product of d2 and a1
Delta1 = d2' * a1; % 25x401
regD1 = (lambda / m) * Theta1; %25x401
regD1 = [zeros(rows(regD1),1) regD1(:,2:end)];

Theta1_grad = (Delta1 / m) + regD1;
Theta2_grad = (Delta2 / m) + regD2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
