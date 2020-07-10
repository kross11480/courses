function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_set = [.01; .03 ; .1 ; .3; 1 ; 3 ; 10 ; 30];
sigma_set = [.01; .03 ; .1 ; .3; 1 ; 3 ; 10 ; 30];
C = 1;
sigma = .1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

%for i = 1:8
%  for j = 1:8
%    C = C_set(i);
%    sigma = sigma_set(j);
%    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%    predictions = svmPredict(model, Xval);
%    temp = predictions - yval;
%    prediction_error(i,j) = norm(temp,2)
%   endfor
% endfor
 
 %[minval, row] = min(min(prediction_error,[],2));
 %[minval, col] = min(min(prediction_error,[],1));
 %C = C_set(ix(row));
 %sigma = sigma_set(col);
% =========================================================================
end
