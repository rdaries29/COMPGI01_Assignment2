%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 2
% Author : Russel Daries,  Nitish Mutha
% Student ID: 16079408 ,15113106 
% Question: 3
% Section: Error bars and fitting
% Description: Using Gaussian Kernel 
% ------------------------------------------------------------------%
close all
clear all
clc

%% 3a

d = 2;
n = 500;
v = 5;
sigma = 0;

% Initialize
e = mvnrnd(zeros(n,1), sigma^2);
x = mvnrnd(zeros(n,d), eye(d,d));
u = randn([n,1]);

K_true = zeros(size(x,1),size(x,1));

% Constructing Kernel Matrix
for l=1:n
    for k=1:n
        K_true(l,k) = exp(-(norm(x(l,:)-x(k,:),2))^2/(2*v^2));
    end
end

%     L = chol(A,'lower') uses only the diagonal and the lower triangle
%     of A to produce a lower triangular L so that L*L' = A.  If
%     A is not positive definite, an error message is printed.  When
%     A is sparse, this syntax of chol is typically faster.

% Matrix badly scaled, adding small amount to diagonals. Just a computation
% limitation
calc_err = 1e-6;
scale_matrix = 1e-6*eye(n,n);
L =  chol(K_true+scale_matrix,'lower');

Y = L*u + e;
% Y = L*u ;


% Function to plot resultant Gaussian Kernel
[xq,yq] = meshgrid(min(x(:,1)):.2:max(x(:,1)), min(x(:,2)):.2:max(x(:,2)));
vq = griddata(x(:,1),x(:,2),Y,xq,yq);
figure;
mesh(xq,yq,vq);

hold on;
plot3(x(:,1),x(:,2),Y,'o')
axis tight;

%% 3b.

n = 500;
d = 10;
v_0 =5;
sigma = 1.3;

train_perc = 0.125;
valid_perc = 0.125;
test_perc = 0.75;
lambda = sigma^2;

X = mvnrnd(zeros(n,d), eye(d,d));
e = mvnrnd(zeros(n,1), sigma^2);

a_exp = -2:0.5:2;
a = 10.^a_exp;

v_vec = a.*v_0;





