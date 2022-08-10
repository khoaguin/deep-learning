function J = costFunctionJ(X, y, theta)

% X: matrix contains training examples
% y: the class labels

m = size(X, 1); % number of training examples
predictions = X*theta; % predictions of hypothesis on all m examples
sqrErrors = (predictions - y).^2; % squared errors

J = 1/(2*m) * sum(sqrErrors);