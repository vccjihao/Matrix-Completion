clear all
close all
clc
addpath('setsparse');
rng('default')
format long
%% Load the original matrix
M3 = imread('c:\MYSVT\juliabw.jpg');
M = double(M3(:,:,2));

% resize the matrix if larger cases needed
[n1,n2] = size(M);
ratio = 1;   
M = imresize(M,[n1*ratio n2*ratio]);
[n1,n2] = size(M);


%% Set parameters
tau = 100000;    % regularization parameter tau
delta = 1;       % step size
maxiter = 1000;   % maximun number of iterations in SVT
tol = 1e-02;     % convergence threshold of SVT
percent = 0.2;   % specifying percentage of samples

disp(['number of rows in image (n1): ',num2str(n1)])
disp(['number of columns in image (n2): ',num2str(n2)])
disp(['regularization parameter (tau): ',num2str(tau)])
disp(['percentage of samples (percent): ',num2str(percent)])
disp(['step size (delta): ',num2str(delta)])
disp(['convergence threshold of SVT (tol): ',num2str(tol)])
disp(['maximun number of iterations in SVT (maxiter): ',num2str(maxiter)])


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


%% display the image of samples only (for comparison)
[x, y] = ind2sub([n1,n2], Omega);
Y = ones(n1,n2)*200;
for i = 1:m
    Y(x(i),y(i)) = M(x(i),y(i));
end
imshow(uint8(Y)); title('Samples')

% display the image of samples only (for comparison)
subplot(1,3,1)
imshow(uint8(M)); title('Original')
subplot(1,3,2)
imshow(uint8(Y)); title([num2str(percent*100),'% Samples'])
subplot(1,3,3)
imshow(uint8(X1)); title('Original SVT')