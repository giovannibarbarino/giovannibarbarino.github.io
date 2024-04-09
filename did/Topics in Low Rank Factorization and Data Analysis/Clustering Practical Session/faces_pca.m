function [mu,U,Y] = faces_pca(X,k)
%% FACES_PCA given a matrix X with data as columns,
% compute the PCA through SVD. It returns a vector
% mu and matrices U,Y such that X = mu + UY, U is
% orthogonal and Y has zero mean rows. If a rank 
% k is specified, returns only the principal k
% components.
mu = mean(X,2); 
if nargin < 2   
    [U,S,V] = svd(X - mu); 
else
    [U,S,V] = svds(X - mu,k);
end 
Y = S*V';


end
