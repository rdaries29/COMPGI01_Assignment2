%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 2
% Author : Russel Daries, Nitish Mutha
% Student ID: 16079408, 15113106
% Question: 1
% Section: Linear Example
% Description: Ridge Regression with Linear Kernel
% ------------------------------------------------------------------%

close all
clear all
clc

%% 1a.

X_gen = randn([100,10]);
w_gen = randn([10,1]);
sigma = 0.1;
lambda = 0.01;

y_true = (X_gen*w_gen)+(randn([100,1])*sigma);
y_true_train = y_true(1:80);
y_true_test = y_true(81:end);

X_gen_train = X_gen(1:80,:);
X_gen_test = X_gen(81:end,:);

w_learn = linear_ridge_regression(X_gen_train,y_true_train,lambda);
y_pred_primal = X_gen_test*w_learn;

% Plot for Train Output vs Test Output
figure;
plot(y_true_test,'r')
hold on
plot(y_pred_primal,'b--*')
xlabel('x','FontSize',15)
ylabel('y','FontSize',15)
set(gca,'fontsize',17);
grid on
set(gcf, 'Color', 'w');
leg=legend('y_{true}','y_{pred_{primal}}','Location','Best')
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('y_out_primal_q1a','-depsc')
close all;

% Prediction and True error
pred_err = mean_square_error(y_true_test,y_pred_primal)

% %Plot of learned and true weight vectors
figure;
plot(w_gen-w_learn,'ro')
xlabel('Indices of w','FontSize',15)
ylabel('w_{true}-w_{learnt}','FontSize',15)
set(gca,'fontsize',17);
grid on
set(gcf, 'Color', 'w');
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('w_difference','-depsc')
close all;

%Weight vector error
w_err = mean_square_error(w_gen,w_learn);

%% 1b.

alpha_learn_dual = linear_ridge_regression_dual(X_gen_train,y_true_train,lambda);
y_pred_dual = X_gen_test*X_gen_train'*alpha_learn_dual;

figure;
plot(y_true_test,'r')
hold on
plot(y_pred_primal,'b--*')
hold on
plot(y_pred_dual,'k:s')
xlabel('x','FontSize',15)
ylabel('y','FontSize',15)
set(gca,'fontsize',17);
grid on
set(gcf, 'Color', 'w');
leg=legend('y_{true}','y_{pred_{primal}}','y_{pred_{dual}}','Location','Best')
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('y_out_dual_q1b','-depsc')
close all;

mse_obs_dual = mean_square_error(y_true_test,y_pred_dual)
mse_pred_dual = mean_square_error(y_pred_primal,y_pred_dual)
