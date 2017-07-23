function n = noise_function(mu,sigma)
z = randn;
x = z*sigma+mu;
n = x;
