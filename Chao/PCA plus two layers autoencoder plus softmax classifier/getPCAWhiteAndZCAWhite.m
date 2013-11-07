function [ xTildeZCAwhite,xTildePCAwhite,xTilde,k,U,S ] = getPCAWhiteAndZCAWhite( x,retainRate )
%GETPCAWHITEANDZCAWHITE Summary of this function goes here
%   Detailed explanation goes here
% Step 0b: Zero-mean the data (by row)
avg=mean(x,1); % Compute the mean pixel intensity value separately for each patch.
x=x-repmat(avg,size(x,1),1);



%calculate eigenvector and eigenvalue
sigma=x*x'/size(x,2);
[U,S,V]=svd(sigma);


% Step 2: Find k, the number of components to retain
%  Write code to determine k, the number of components to retain in order
%  to retain at least 99% of the variance.
if retainRate==1
    k=size(x,1);
else
    k = 1; % Set k accordingly
    L=diag(S);
    shreshold=retainRate*sum(L);
    temp=0;
    while(true)
        if(temp>=shreshold) 
            break;
        end   
        temp=temp+L(k);
        k=k+1;
    end
end



epsilon=10^-5;

xRot = U' * x; 
xTilde = U(:,1:k)' * x; % reduced dimension representation of the data,

xPCAwhite = diag(1./sqrt(diag(S) + epsilon)) * xRot;
xTildePCAwhite = diag(1./sqrt(diag(S(1:k,1:k)) + epsilon)) * xTilde;

xZCAwhite = U * xPCAwhite;
xTildeZCAwhite = U(1:k,1:k) * xTildePCAwhite;

end

