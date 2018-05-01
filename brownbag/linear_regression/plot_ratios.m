% Demonstrate plotting of the ratio of a radius of a sphere to it's diameter, circumference, and volume

X = 0:0.1:2;

y_diameter = X .* 2;
y_circumference = y_diameter .* pi;
y_volume = 4/3 * pi * (X .^ 3);

y_limit = 120;
x_limit = 30;

figure;
hold on;
axis([0 x_limit 0 y_limit]);
set(gca, 'XTick', [0:1:x_limit]);
%set(gca, 'YTick', [0:10:y_limit]);
%plot(X, y_diameter,'b');
%plot(X, y_diameter, "xr");

%plot(X, y_circumference,'g');
%plot(X, y_circumference, "xr");

plot(X .^ 3, y_volume,'m');
plot(X .^ 3, y_volume, "xr");

xlabel("len^3");
%title("Simple Mathmatical Relationships");
hold off;
