function weights = linear_ridge_regression_dual(X,y,lambda)

weights = (X*X' + (lambda*eye(size(X,1))))\y;
