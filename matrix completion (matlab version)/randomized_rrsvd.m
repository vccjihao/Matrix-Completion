% This code is for testing a rank revealing randomized algorithm for large scale
% matrix completion problems

% We acknowledge support from RSCA (Research, Scholarship, and Creative
% Activities Program) 2016-2017 at Cal Poly Pomona

% [U,D,V]=randomized_rrsvd(A,k,tau,p,np,doReuse,preV,r)
%     A rank revealing randomized singular value decomposition algorithm
%     for adaptively finding low-rank approximations with singular values
%     greater than the threshold tau.
%
% Inputs:
%   A - the original matrix.
%   k - initial guess of the target rank
%   tau - scala parameter, which is the Lagrange multiplier trading off
%         between the nuclear and Frobenius norm in the SVT algorithm.
%   p - oversampling number.
%   np - number of powers.
%   doReuse - if doReuse is 1, then r3svd uses the range space of preV
%             to sample the range space of A. Otherwise, r3svd creates a new
%             Gaussian matrix to sample A.
%   preV - the matrix to sample A when doReuse is 1
%   deltak -  rank increase per step
%
% Outputs:
%   U - the orthogonal matrix containing left singular vectors.
%   D - the diagonal matrix containing left singular vectors..
%   V - "the matrix containing right singular vectors".
%
% Reference:
%    H. Ji, W. Yu, and Y. Li,
%    A rank revealing randomized singular value decomposition (r3svd)
%    algorithm for low-rank matrix approximations, Computing Research
%    Repository arXiv:1605.08134 (2016) 1-10.
%
% Written by: Hao Ji
% Email: hji@cpp.edu

function [U,D,V]=randomized_rrsvd(A,k,tau,p,np,doReuse,preV,deltak)

[rows,cols] = size(A);
U = [];
V = [];
D = [];

i = 0;
while(1)
    i = i+1;
    
    if i == 1
        
        % -- set the sampling matrix  -- %
        if(doReuse == 1)
            sPreV = size(preV,2);
            Omega = [preV,randn(cols,k-sPreV+p)]; % use a specified matrix preV
        else
            Omega = randn(cols,k+p);    % use Gaussian matrix
        end
        Y = A*Omega;
        
        % -- the power scheme -- %
        for j=1:np
            Y = A*(A'*Y);
        end
        [Q,R] = qr(Y,0);
        
        % -- approximate the singular components -- %
        B = Q'*A;
        [Ub,Db,Vb] = svd(B,'econ');
        protoU = Q*Ub(:,1:k);
        protoD = diag(Db(1:k,1:k));
        protoV = Vb(:,1:k);
        
    else
        
        % -- set the sampling matrix  -- %
        Omega = randn(cols,deltak+p);
        Y = A*Omega;

        % -- the power scheme  -- %
        for j=1:np
            Y = A*(A'*Y);
        end
        [Q,R] = qr(Y,0);
        
        % -- approximate the singular components -- %
        B = Q'*A-Q'*(U*(U'*A));               % orthogonalization Process
        [Ub,Db,Vb] = svd(B,'econ');
        protoU = Q*Ub(:,1:deltak);
        protoD = diag(Db(1:deltak,1:deltak));
        protoV = Vb(:,1:deltak);
    end
    
    idx = find(protoD >= tau,1,'last');    % check if there is any components with singular values greater than the threshold tau
    if(isempty(idx))
        break;
    else
        V = [V,protoV(:,1:idx)];
        U = [U,protoU(:,1:idx)];
        D = [D;protoD(1:1:idx)];
    end
    
end
D = diag(D);
