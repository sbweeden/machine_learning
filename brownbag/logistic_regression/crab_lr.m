% crab sex typing by logistic regression

data = load('crabdata.txt');
m = length(data);
n = size(data,2)-1;

X = data(:, 1:n);
y = data(:, n+1);

%females = X(y==0,:);
%males = X(y==1,:);

%figure;
%hold on;
%plot(females(:,1), females(:,2), "ro");
%plot(males(:,1), males(:,2), "b+");
%hold off;


plotData(X, y);

% Put some labels
hold on;

% Labels and Legend
xlabel('Carapace Size')
ylabel('Claw Size')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;

%
% Now find a linear decision boundary
%
numPolynomial = 1; % set to 1 for a straight line
X_mapped = mapFeature(X(:,1), X(:,2), numPolynomial);

% Initialize fitting parameters
initial_theta = zeros(size(X_mapped, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X_mapped, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X_mapped, y, numPolynomial);
hold on;
title("Decision Boundary");

% Labels and Legend
xlabel('Carapace Size');
ylabel('Claw Size');

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X_mapped);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

