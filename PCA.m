function [ W ] = PCA( F, reduced_dimensions )
%PCA 
%   Detailed explanation goes here
mu = mean(F);
MU = ones(size(F))*diag(mu);

X = F - MU;

[eigenVectors, eigenValues] = eig(X' * X);

shape = size(eigenVectors);
num_cols = shape(2);

W = eigenVectors(:, (num_cols-reduced_dimensions + 1):num_cols);
end

