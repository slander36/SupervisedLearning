function [ xTildeZCAwhite,xTildePCAwhite,xTilde ] = processTestData( x,U,k,S )
%PROCESSTESTDATA Summary of this function goes here
%   Detailed explanation goes here


avg=mean(x,1); % Compute the mean pixel intensity value separately for each patch.
x=x-repmat(avg,size(x,1),1);


epsilon=10^-5;

xRot = U' * x; 
xTilde = U(:,1:k)' * x; % reduced dimension representation of the data,

xPCAwhite = diag(1./sqrt(diag(S) + epsilon)) * xRot;
xTildePCAwhite = diag(1./sqrt(diag(S(1:k,1:k)) + epsilon)) * xTilde;

xZCAwhite = U * xPCAwhite;
xTildeZCAwhite = U(1:k,1:k) * xTildePCAwhite;
end

