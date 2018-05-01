% sphere_lr.m

% Shows how lr applied to the volume of a sphere works, when polynomials are added to the features.

X = [0:0.1:2]';
y_volume = 4/3 * pi * (X .^ 3);
predictFor=3;


% First visualize with no polynomials
visualizeLinearReg(X, y_volume, [], predictFor, "Fitting linear regression to radius");

% Now visualize with r and r^2 as features
visualizeLinearReg(X, y_volume, [2], predictFor, "Fitting linear regression to [r, r^2]");

% Now visualize with r, r^2 and r^3 as features
visualizeLinearReg(X, y_volume, [2 3], predictFor, "Fitting linear regression to [r, r^2, r^3]");
