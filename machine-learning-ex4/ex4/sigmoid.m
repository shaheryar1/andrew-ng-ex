function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g= zeros(size(z));
s=size(z);
    for j=1:s(1)
        for i=1:s(2)
           g(j,i)=1/(1+exp(1)^(-z(j,i)));
        end
    end    
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================

end