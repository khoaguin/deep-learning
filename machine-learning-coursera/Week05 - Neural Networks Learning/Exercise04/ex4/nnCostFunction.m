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
m = size(X, 1); % m is the number of rows of X, which is 5000
         
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

a1 = [ones(m, 1) X]; % adding the column of 1?s to the X matrix
num_layers = 3; % number of layers
A{1} = a1; % A contains all of the activation of our layers
% Theta contains all the parameters
Theta{1} = Theta1;
Theta{2} = Theta2;

for i = 2:num_layers
    A{i} = sigmoid(A{i-1} * (Theta{i-1})');
    if i ~= num_layers
        A{i} = [ones(size(A{i-1}, 1), 1) A{i}];
    end
end

h = A{num_layers}; % h will be A{3} which is the activation of the last layer

y_matrix = zeros(m, num_labels); % y_matrix will be 5000x10

for i = 1:m
    y_matrix(i,y(i)) = 1;
end

J = -(1/m) * sum(sum(y_matrix .* log(h) + (1-y_matrix) .* log(1-h)));

% Now we do Regularization
theta1_not1 = Theta1(:, 2:end); % we dont regularized the 1st collumn
theta2_not1 = Theta2(:, 2:end);

regularization = lambda/(2*m) * (sum(sum(theta1_not1 .^ 2)) + ...
                                sum(sum(theta2_not1 .^ 2)));
                            
J = J + regularization;

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
for t = 1:m
    % Set the input layer?s values (a(1)) to the t-th training example x(t)
    a1 = [1; X(t, :)'];
    % Computing the activations (a(2), a(3)) for layers 2 and 3
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    % Finding Delta3
    yk = ([1:num_labels] == y(t))';
    Delta3 = a3 - yk;
    
    % Finding Delta2
    Delta2 = Theta2' * Delta3 .* sigmoidGradient([1; z2]);
    Delta2 = Delta2(2:end);
    
    Theta1_grad = Theta1_grad + Delta2 * a1';
    Theta2_grad = Theta2_grad + Delta3 * a2';
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) .* Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) .* Theta2(:, 2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
