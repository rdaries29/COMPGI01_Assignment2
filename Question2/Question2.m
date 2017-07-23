%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 2
% Author : Russel Stuart Daries, Nitish Mutha
% Student ID: 16079408 ,15113106
% Question: 2
% Section: Non-Linear Kernel
% Description: Ridge Regression with Non-Linear Kernel
% ------------------------------------------------------------------%

close all
clear all
clc

%% 2a.
d = 2;
sigma = 0;
sample_length = 10000;
x = randn([sample_length,d]);
% x = mvnrnd(zeros(sample_length,d), eye(d,d));

%Calculation for z
for i=1:sample_length
    temp_result = x(i,:)'*x(i,:);
    z(i,:) = reshape(temp_result,[d^2,1]);
end

% Calculation for w
w = randn([d^2,1]);

y = z*w;

% % Plot for Quadratic Relationship
figure;



plot3(x(:,1),x(:,2),y,'*')
set(gcf, 'Color', 'w');
xlabel('x_{1}','FontSize',15)
ylabel('x_{2}','FontSize',15)
zlabel('y','FontSize',15)
% set(gca,'fontsize',17);
grid on
% set(gcf, 'Color', 'w');
leg=legend('y_{data points}','Location','BestOutside')
set(leg,'FontSize',15)
hold on;
[xq,yq] = meshgrid(min(x(:,1)):.2:max(x(:,1)), min(x(:,2)):.2:max(x(:,2)));
vq = griddata(x(:,1),x(:,2),y,xq,yq);
mesh(xq,yq,vq);

set(gca,'YMinorTick','on')
grid minor
axis tight;
print('y_out_quad_q2a','-depsc')
close all;

%% 2b.

d = 20;
n = 500;
sigma = 5;
lambda = sigma^2;
sample_length = 500;

% Generated data
X_gen = randn([n,d]);
% X_gen = mvnrnd(zeros(sample_length,d), eye(d,d)); 

w_gen = randn([d,1]);

% Splitting Data
X_train = X_gen(1:round(n*0.25),:);
X_test = X_gen(round(n*0.25)+1:end,:);

for i=1:sample_length
    temp_result = X_gen(i,:)'*X_gen(i,:);
    z_2b(i,:) = reshape(temp_result,[d^2,1]);
end

w_gen_kernel = randn([d^2,1]);

y_true = (z_2b*w_gen_kernel)+(mvnrnd(zeros(sample_length,1), sigma^2));
y_true_train = y_true(1:round(n*0.25));
y_true_test = y_true(round(n*0.25)+1:end);


% Calculating alphas
alphas_learn_dual = non_linear_regression_quad(X_train,y_true_train,lambda);

y_pred = zeros(size(X_test,1),1);
% Calculation for predicted y
for i=1:size(X_test,1)
    temp = 0;
    for j=1:size(X_train,1)
        temp  = temp + alphas_learn_dual(j)*kernel_map(X_train(j,:),X_test(i,:),2);
    end
    y_pred(i) =  temp;
end

figure;
plot(y_true_test,'ro')
hold on
plot(y_pred,'b+')
hold on
xlabel('Test vectors of x','FontSize',15)
ylabel('Output of y','FontSize',15)
set(gca,'fontsize',17);
grid on
set(gcf, 'Color', 'w');
leg=legend('y_{true}','y_{pred}','Location','Best')
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('y_out_quad_q2b','-depsc')
grid on
close all;

y_err = mean_square_error(y_true_test,y_pred)




%% 2c.

% Part 1 of Question 2c
% Splitting Data for Cross Validation
X_train = X_gen(1:round(n*0.25),:);
X_valid = X_gen(round(n*0.25)+1:round(n*0.40),:);
X_test = X_gen(round(n*0.40)+1:end,:);

y_true_train_2c = y_true(1:round(n*0.25));
y_true_valid_2c = y_true(round(n*0.25)+1:round(n*0.40));
y_true_test_2c = y_true(round(n*0.40)+1:end);

a_exp = -4:0.25:4;
a = 10.^a_exp;

lambda_vec = a.*sigma^2;

for i=1:size(lambda_vec,2)
    alphas_learn(:,i) = non_linear_regression_quad(X_train,y_true_train_2c,lambda_vec(i));
end

% Calculation for predicted y based on training and compared to validation
% set
for m=1:size(lambda_vec,2)
    for i=1:size(X_valid,1)
        temp = 0;
        for j=1:size(X_train,1)
            temp  = temp + (alphas_learn(j,m) * kernel_map(X_train(j,:),X_valid(i,:),2));
        end
        y_pred_lambda(i) =  temp;
    end
    y_err_valid(m) = mean_square_error(y_true_valid_2c,y_pred_lambda)
end

figure;
plot(lambda_vec,y_err_valid,'r*-')
xlabel('lambda','FontSize',15)
ylabel('Prediction error of y','FontSize',15)
set(gca,'fontsize',17);
grid on
set(gcf, 'Color', 'w');
% leg=legend('y_{true}','y_{pred_{primal}}','y_{pred_{dual}}','Location','Best')
% set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('y_err_2c_a','-depsc')
grid on
close all;

[number,index] = min(y_err_valid);
min_lambda = lambda_vec(index)
min_a = a(index)

% Part 2 of Question 2c
% 100 Trials 

number_of_trials = 100;
sigma = 5;

for i=1:number_of_trials
    % Generated data
    feature_length = 20;
    sample_length = 500;
    X_gen_trial = randn([sample_length,feature_length]);
    w_gen_trial = randn([feature_length,1]);

    X_train_trial = X_gen_trial(1:round(sample_length*0.25),:);
    X_valid_trial = X_gen_trial(round(sample_length*0.25)+1:round(sample_length*0.40),:);
    X_test_trial = X_gen_trial(round(sample_length*0.40)+1:end,:);
    
    for mm=1:sample_length
        temp_result = X_gen_trial(mm,:)'*X_gen_trial(mm,:);
        z_2c(mm,:) = reshape(temp_result,[feature_length^2,1]);
    end

    w_gen_kernel = randn([feature_length^2,1]);

    y_true_trial = (z_2c*w_gen_kernel)+(mvnrnd(zeros(sample_length,1), sigma^2));

    y_true_train_trial = y_true_trial(1:round(sample_length*0.25));
    y_true_valid_trial = y_true_trial(round(sample_length*0.25)+1:round(sample_length*0.40));
    y_true_test_trial = y_true_trial(round(sample_length*0.40)+1:end);    

    for k=1:size(lambda_vec,2)
        alphas_learn_trial(:,k) = non_linear_regression_quad(X_train_trial,y_true_train_trial,lambda_vec(k));
    end

    for m=1:size(lambda_vec,2)
        for n=1:size(X_valid_trial,1)
            temp = 0;
            for j=1:size(X_train_trial,1)
                temp  = temp + (alphas_learn_trial(j,m) * kernel_map(X_train_trial(j,:),X_valid_trial(n,:),2));
            end
            y_pred_lambda_trial(n) =  temp;
        end
        if (isnan(y_pred_lambda_trial))
            fprintf('NaN calculated')
        end
        y_err_valid_trial(m) = mean_square_error(y_true_valid_trial,y_pred_lambda_trial);
    end
   
    [mse_number,mse_index] = min(y_err_valid_trial);
    min_mse_index(i) = mse_index;
    min_mse_vec_valid_set(i) = mse_number;
    min_mse_lambda_vec(i) = lambda_vec(mse_index);
    min_mse_a_vec(i) = a(mse_index);
    
    % Creating alpha with the best lambda from validation test
    alphas_learn_optimal_lambda = non_linear_regression_quad(X_train_trial,y_true_train_trial,min_mse_lambda_vec(i));
    for nn=1:size(X_test_trial,1)
        temp = 0;
        for jj=1:size(X_train_trial,1)
            temp  = temp + (alphas_learn_optimal_lambda(jj) * kernel_map(X_train_trial(jj,:),X_test_trial(nn,:),2));
        end
        y_pred_lambda_test(nn) =  temp;
    end
    
    y_err_test_trial(i) = mean_square_error(y_true_test_trial,y_pred_lambda_test);
    
end

% Resultant MSE calculations
mean_mse_valid = mean(min_mse_vec_valid_set)
mean_mse_test = mean(y_err_test_trial)
mean_lambda = mean(min_mse_lambda_vec)


figure;
plot(min_mse_vec_valid_set,'r-*')
hold on
plot(mean_mse_valid*ones(length(min_mse_vec_valid_set)),'b--')
ylabel('mse for validation')
grid on;
close all;

figure;
plot(y_err_test_trial,'r*')
hold on
plot(mean_mse_test*ones(length(y_err_test_trial)),'b--')
xlabel('Trial number','FontSize',15)
ylabel('Prediction Error of y','FontSize',15)
grid on;
set(gcf, 'Color', 'w');
leg=legend('pred_{err}','pred_{avg}','Location','Best')
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('y_err_2c_b','-depsc')
close all;

figure;
plot(min_mse_lambda_vec,'ko')
hold on
plot(mean_lambda*ones(length(min_mse_lambda_vec)),'b--')
xlabel('Trial number','FontSize',15)
ylabel('Optimal \lambda','FontSize',15)
grid on;
set(gcf, 'Color', 'w');
leg=legend('\lambda_{optimal}','\lambda_{avg}','Location','Best')
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('y_err_2c_c','-depsc')
close all;

% figure;
% hist(min_mse_index,25)
% grid on;
% close all;















