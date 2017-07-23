function weights = linear_ridge_regression(X,y,lambda)


weights = ((X'*X + (lambda*eye(size(X,2))))^-1)*X'*y;

