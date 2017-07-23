function error = mean_square_error(y_true,y_pred)

error = 0;
n = length(y_true);
for i=1:n
    error = error + (y_true(i) - y_pred(i))^2;
end
error = (1/n)* error;
