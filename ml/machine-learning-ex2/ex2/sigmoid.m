function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


one_vec = ones(size(z));
neg_one_vec = -1 .* one_vec;
g = one_vec ./ (one_vec + exp(neg_one_vec.*z)); 


% =============================================================

end
