function alphas = non_linear_regression_quad(X,y,lambda)

[dim1,dim2] = size(X);

K = zeros(dim1,dim1);

for m=1:dim1
    for n=1:dim1
        K(m,n) = kernel_map(X(m,:),X(n,:),2);
    end
end

alphas = (K + (lambda * eye(dim1,dim1)) )\y;
