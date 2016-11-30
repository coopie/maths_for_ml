function [ X ] = center( F )
% centers a matrix of shape (features X examples)
mu = mean(F, 2);

MU = repmat(mu, 1, size(F, 2));

X = F - MU;
end

