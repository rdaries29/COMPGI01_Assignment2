function k = kernel_map(x_train,x_test,power)

[dim_1,dim2] = size(x_train);

train_vec_map = reshape((x_train'*x_train),[dim2^power,1]);
test_vec_map = reshape((x_test'*x_test),[dim2^power,1]);

k =  train_vec_map'* test_vec_map ;

