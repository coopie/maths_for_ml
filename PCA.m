function [ U, inv_root_lambda ] = PCA( F, reduced_dimensions )
%PCA 
%   Matrix is of shape (features X examples)
mu = mean(F, 2);

MU = repmat(mu, 1, size(F, 2));

X = F - MU;
% X
[eigenVectors, eigenValues] = eig(X' * X);
%

% some reshaping so eigenVectors and eigenValues are ordered by eigenvalue descending
eigenVectors = fliplr(eigenVectors);
eigenValues = flipud(fliplr(eigenValues));


% return inv_root_lambda so whitened pca can be used as well
inv_root_lambda = eigenValues ^ (-0.5);
U = X * eigenVectors * inv_root_lambda;

end




