% gameshownn.m

%
% Uses a neural network to learn whether or not a contestant is going to win the game show
%

%
% Loads game show observed samples from a csv file. Each sample is:
% prizeDoor,contestantDoor,hostDoor,swapFlag,winnerFlag
%
% It turns out the hostDoor is irrelevant and can be omitted as a feature and the NN will work all the same
%
data = load('gameshowdata.txt');
totalM = length(data);
n = size(data,2)-1;

% create training, cross-validation, and test sets. Training is 60% of samples, crossval is 20% and test is the rest (approx 20%)
m_train = floor(totalM * 1); % .6
%m_crossval = floor(totalM * .2);
%m_test = totalM - m_train - m_crossval;

X_train = data(1:m_train, 1:n);
%X_crossval = data(1+m_train:(m_train+m_crossval), 1:n);
%X_test = data(1+m_train+m_crossval:totalM, 1:n);

y_train = data(1:m_train, n+1);
%y_crossval = data(1+m_train:m_train+m_crossval, n+1);
%y_test = data(1+m_train+m_crossval:totalM, n+1);

% Setup NN parameters
input_layer_size  = n;  % number of features
hidden_layer_size = 10;  % complete guess - n worked
num_labels = 2;         % 2 labels, either winner or not winner



fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];




options = optimset('MaxIter', 100);

%  You should also try different values of lambda
lambda = 0.02;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred_train = predictnn(Theta1, Theta2, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train == y_train)) * 100);

%pred_crossval = predictnn(Theta1, Theta2, X_crossval);
%fprintf('\nCross validation Set Accuracy: %f\n', mean(double(pred_crossval == y_crossval)) * 100);

%pred_test = predictnn(Theta1, Theta2, X_test); 
%fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);
