function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
a1 = [ones(size(X, 1), 1) X]; % adding bias unit to X, a1 is 5000x401
z2 = a1 * Theta1'; % Theta1 is 25x401
a2 = sigmoid(z2); % a2 is 5000x25
% fprintf('\nsize of a2 is %f\n', size(a2, 1));

a2 = [ones(size(a2, 1), 1) a2]; 
z3 = a2 * Theta2'; % Theta2 is 10x26
h = sigmoid(z3); % h is 5000 x 10
% size(h); 
% fprintf('\nsize of h is %f\n', size(h));

[max_value, p] = max(h, [], 2); % p is the indices of max values in h
% =========================================================================


end
