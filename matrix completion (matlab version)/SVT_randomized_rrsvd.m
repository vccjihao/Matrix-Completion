% This code is for testing a rank revealing randomized algorithm for large scale
% matrix completion problems

% We acknowledge support from RSCA (Research, Scholarship, and Creative
% Activities Program) 2016-2017 funding support at Cal Poly Pomona

function [U,Sigma,V,numiter,out]  = SVT_randomized_rrsvd(n,Omega,b,tau,delta,maxiter,tol,p,np)

global VERBOSE
if isempty(VERBOSE)
    % -- feel free to change these 'verbosity' parameters
    % VERBOSE = false;
    VERBOSE = 1;    % a little bit of output
    % VERBOSE = 2;    % even more output
end

time1 = cputime;
%% set parameters
n1 = n(1);
n2 = n(2);
m = length(Omega);
incre = 5;
out.residual = zeros(maxiter,1);
out.rank= zeros(maxiter,1);
out.time = zeros(maxiter,1);
out.nuclearNorm = zeros(maxiter,1);
out.p = zeros(maxiter,1);

%% update matrix Y which is used in the 1st SVT iteration
[i, j] = ind2sub([n1,n2], Omega);
Y = sparse(i,j,b,n1,n2,m);
normProjM = normest(Y,1e-2);
k0 = ceil(tau/(delta*normProjM));
normb = norm(b);
y = k0*delta*b; % kicking by k0 steps
Y = setsparse(Y, i, j, y);

%% start SVT iterations
r = 0;
V = [];
relResMin = 1000000;
if VERBOSE==1, fprintf('\nIteration:   '); end
for k = 1:maxiter,
    if VERBOSE==1, fprintf('\b\b\b\b%4d',k);  end
    s = r + incre;
    
    % perform R3SVD to obtain a low-rank approximation with singular values
    % greater than tau
    if k == 1
        [U,Sigma,V] = r3svd(Y,s,tau,p,np,0,[],[]);    % R3SVD starts from a Gaussian matrix
    else
        [U,Sigma,V] = r3svd(Y,s,tau,p,np,1,V,incre);  % R3SVD reuses the previous matrix V in the subsequent iterations 
    end
    
    % perform the singular value thresholding
    sigma = diag(Sigma);
    r = sum(sigma > tau);
    U = U(:,1:r);
    V = V(:,1:r);
    sigma = sigma(1:r) - tau;
    Sigma = diag(sigma);
    
    % get elements on the sample locations from the completed matrix 
    x = XonOmega(U*diag(sigma),V,Omega);
    
    % keep track of err, time, rank, nuclearNorm, and oversampling number
    eTime = cputime - time1;
    if VERBOSE == 2
        fprintf('iteration %4d, rank is %2d, rel. residual is %.1e\n',k,r,norm(x-b)/normb);
    end
    relRes = norm(x-b)/normb;
    out.residual(k) = relRes;
    out.time(k) = eTime;
    out.rank(k) = r;
    out.nuclearNorm(k) = sum(sigma);
    out.p(k) = p;

    % check convergence
    if (relRes < tol)
        break
    end
    if (norm(x-b)/normb > 1e5)
        disp('Divergence!');
        break
    end
    
    % reset the time clock
    time1 = cputime;
    
    % update power number or oversampling number for the next SVT iteration
    if relRes > relResMin*(1+1e-2)
        %     if relRes > relResMin
        if np <= 5
            np = np +1;
        else
            p = p + 4;
        end
        %         np = np +1;
        relResMin = relRes;
    else
        relResMin = relRes;
    end
    
    % update matrix Y
    y = y + delta*(b-x);
    Y = setsparse(Y, i, j, y);
    
end

if VERBOSE==1, fprintf('\n'); end
numiter = k;
out.residual = out.residual(1:k,:);
out.time = out.time(1:k,:);
out.rank= out.rank(1:k,:);
out.nuclearNorm= out.nuclearNorm(1:k,:);
out.p = out.p(1:k,:);