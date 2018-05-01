function a2 = shane(X, Theta1)
	a2 = zeros(3, 1);

	for i = 1:1
		for j = 1:3
			a2(i) = a2(i) + X(j) * Theta1(i,j);
		end
		a2(i) = sigmoid(a2(i));
	end
end
