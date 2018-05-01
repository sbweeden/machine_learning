function visualizeLinearReg(X, y, polyNomials, predictFor,titleText)
% Visualize the training data and linear regression applied to it
% for this to visualize nicely, predictFor should be > max(X) since we plot 
% an extrapolation from the training data up to the predictFor value.

% First plot X against y to visualize the data
fprintf('Plotting training data\n');
figure;
hold on;
plot(X, y,'m');
plot(X, y, "xr");
xlabel("X");
title(titleText);

fprintf('Program paused. Press enter to continue.\n');
pause;

% Run linear regression to get optimized theta

% Setup the data matrix appropriately, and add ones for the intercept term, and 
% any requested polynomials
[m, n] = size(X);
X_r = [ones(m, 1) X];
X_predictFor = [1 predictFor];
X_extrapolation = [max(X):0.1:predictFor]';
X_extrapolation_r = [ones(size(X_extrapolation, 1),1) X_extrapolation];

for i=1:size(polyNomials,2)
	X_r = [X_r X.^polyNomials(i)];
	X_predictFor = [X_predictFor predictFor^polyNomials(i)];
	X_extrapolation_r = [X_extrapolation_r X_extrapolation.^polyNomials(i)];
end

% Initialize fitting parameters
initial_theta = zeros(size(X_r, (n+1)), 1);

% Set regularization parameter lambda - we use zero because we know
% that in this example we will exactly fit one of the polynomials
lambda = 0;

[theta] = trainLinearReg(X_r, y, lambda);

% Done all the calculations - now visualize

J = linearRegCostFunction(X_r, y, theta, lambda);
fprintf('Computed cost:  %f\n', J);

fprintf('Plotting hypothesis\n');
plot(X, X_r * theta);
fprintf('Program paused. Press enter to continue.\n');
pause;

% predict and extrapolate current hypothesis to visualize the prediction
hypPredictFor = X_predictFor * theta;
fprintf('Predicted y for x=%d is: %f\n', predictFor, hypPredictFor);


hypXextrapolation = X_extrapolation_r * theta;

plot(X_extrapolation, hypXextrapolation, "b--");
plot(predictFor, hypPredictFor, "rx");
fprintf('The theta values are\n');
theta
fprintf('Program paused. Press enter to continue.\n');
pause;
hold off;

end
