function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
%

max_error = Inf;
temp_c    = [0.01 0.03 0.1 0.3 1 3 10 30];
temp_sig  = [0.01 0.03 0.1 0.3 1 3 10 30];
n = size(temp_c);
n=n(1,2);

for i=1:n
    for j=1:n  
        model = svmTrain(X, y, temp_c(1,i), @(x1, x2) gaussianKernel(x1, x2, temp_sig(1,j)));
        prediction = svmPredict(model, Xval);
        prediction_Error = mean(double(prediction ~= yval));
        if prediction_Error < max_error
           max_error = prediction_Error;
           C = temp_c(1,i);
           sigma = temp_sig(1,j);
        end
    end
end


% =========================================================================

end
