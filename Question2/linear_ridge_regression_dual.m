function alphas = linear_ridge_regression_dual(X,y,lambda)

alphas = (X*X' + (lambda*eye(size(X,1))))\y;
