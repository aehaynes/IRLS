from numpy import array, diag, dot, maximum, empty, repeat, ones
from numpy.linalg import inv

def IRLS(y, X, maxiter, d = 0.0001, tol = 0.001):
	n,p = X.shape
	delta = array( repeat(d, n) ).reshape(1,n)
	w = repeat(1, n)
	W = diag( w )
	B = dot( inv( X.T.dot(W).dot(X) ), 
			 ( X.T.dot(W).dot(y) ) )

	for _ in range(maxiter):
		_w = abs(y - X.dot(B)).T
		w = float(1)/maximum( delta, _w )
		W = diag( w[0] )
		B = dot( inv( X.T.dot(W).dot(X) ), 
				 ( X.T.dot(W).dot(y) ) )
	return B


input_str = '''2 7
0.18 0.89 109.85
1.0 0.26 155.72
0.92 0.11 137.66
0.07 0.37 76.17
0.85 0.16 139.75
0.99 0.41 162.6
0.87 0.47 151.77
4
0.49 0.18
0.57 0.83
0.56 0.64
0.76 0.18
'''

output_str ='''105.22
142.68
132.94
129.71
'''

input_list = input_str.split('\n')

p,n = [ int(i) for i in input_list.pop(0).split() ]
X = empty( [n, p+1] )
X[:,0] = repeat( 1, n)
y = empty( [n, 1] )
for i in range(n):
	l = [ float(i) for i in input_list.pop(0).split() ]
	X[i, 1:] = array( l[0:p] )
	y[i] = array( l[p] )

n = [ int(i) for i in input_list.pop(0).split() ][0]
X_new = empty( [n, p] )
for i in range(n):
	l = [ float(i) for i in input_list.pop(0).split() ]
	X_new[i] = array( l[0:p] )






	