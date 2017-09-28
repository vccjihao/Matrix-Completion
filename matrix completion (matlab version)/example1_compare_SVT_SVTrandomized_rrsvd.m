% This code is for testing a rank revealing randomized algorithm for large scale
% matrix completion problems

% We acknowledge support from RSCA (Research, Scholarship, and Creative
% Activities Program) 2016-2017 funding support at Cal Poly Pomona


clear all
close all
clc
addpath('setsparse');
rng('default')
format long
%% Load the original matrix
M3 = imread('2.1.05.tiff');
M = double(M3(:,:,2));

% resize the matrix if larger cases needed
[n1,n2] = size(M);
ratio = 1;   
M = imresize(M,[n1*ratio n2*ratio]);
[n1,n2] = size(M);


%% Set parameters
tau = 100000;    % regularization parameter tau
delta = 1;       % step size
maxiter = 2000;   % maximun number of iterations in SVT
tol = 1e-02;     % convergence threshold of SVT
p = 5;           % oversampling number (used in R3SVD)
np = 1;          % number of powers (used in R3SVD)
percent = 0.2;   % specifying percentage of samples

disp(['number of rows in image (n1): ',num2str(n1)])
disp(['number of columns in image (n2): ',num2str(n2)])
disp(['regularization parameter (tau): ',num2str(tau)])
disp(['percentage of samples (percent): ',num2str(percent)])
disp(['step size (delta): ',num2str(delta)])
disp(['convergence threshold of SVT (tol): ',num2str(tol)])
disp(['maximun number of iterations in SVT (maxiter): ',num2str(maxiter)])
disp(['oversampling number (p): ',num2str(p)])
disp(['number of powers (np): ',num2str(np)])

%% generate random samples from the loaded image
m = floor((n1*n2)*percent);    % number of samples
Omega = randsample(n1*n2,m);   % an array of sample indices
data = M(Omega);               % an array of samples


%% The original SVT algorithm
rng('default')
format long
fprintf('\nSolving by SVT...\n');
tic
[U1,S1,V1,numiter1,out1] = SVT([n1 n2],Omega,data,tau,delta,maxiter,tol);
toc

% construct the completed matrix
X1 = U1*S1*V1';
fprintf('Original: The recovered rank is %d\n',rank(X1) );
fprintf('Original: The relative error on Omega is: %d\n', norm(data-X1(Omega))/norm(data))
fprintf('Original: The relative recovery error is: %d\n', norm(M-X1,'fro')^2/norm(M,'fro')^2)
fprintf('Original: The relative recovery in the spectral norm is: %d\n', norm(M-X1)/norm(M))


%% The modified SVT algorithm based on R3SVD
rng('default')
format long

fprintf('\nSolving by SVT using randomized algorithm...\n');
tic
[U2,S2,V2,numiter2,out2] = SVT_randomized_rrsvd([n1 n2],Omega,data,tau,delta,maxiter,tol,p,np);
toc

% construct the completed matrix
X2 = U2*S2*V2';

% Show results
fprintf('RSVD: The recovered rank is %d\n',rank(X2) );
fprintf('RSVD: The relative error on Omega is: %d\n', norm(data-X2(Omega))/norm(data))
fprintf('RSVD: The relative recovery error is: %d\n', norm(M-X2,'fro')^2/norm(M,'fro')^2)
fprintf('RSVD: The relative recovery in the spectral norm is: %d\n', norm(M-X2)/norm(M))

%% display the image of samples only (for comparison)
[x, y] = ind2sub([n1,n2], Omega);
Y = ones(n1,n2)*200;
for i = 1:m
    Y(x(i),y(i)) = M(x(i),y(i));
end
imshow(uint8(Y)); title('Samples')

% display the image of samples only (for comparison)
subplot(2,2,1)
imshow(uint8(M)); title('Original')
subplot(2,2,2)
imshow(uint8(Y)); title([num2str(percent*100),'% Samples'])
subplot(2,2,3)
imshow(uint8(X1)); title('Original SVT')
subplot(2,2,4)
imshow(uint8(X2)); title('SVT using R3SVD')