# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:36:29 2016

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('C:\\Dropbox\\trading')

import interpolate as intp
import scipy as sp


x = np.linspace(0, 1, 21)
noise = 1e-1*np.random.randn(x.size)
noise = np.array([-0.03298601, -0.08164429, -0.06845745, -0.20718593,  0.08666282,
0.04702094,  0.08208645, -0.1017021 , -0.03031708,  0.22871709,
-0.10302486, -0.17724316, -0.05885157, -0.03875947, -0.1102984 ,
-0.05542001, -0.12717549,  0.14337697, -0.02637848, -0.10353976,
-0.0618834 ])

y = np.exp(x) + noise
pp9 = intp.SmoothSpline(x, y, p=.9)
pp99 = intp.SmoothSpline(x, y, p=.99) #, var=0.01)

y99 = pp99(x); y9 = pp9(x)
np.allclose(y9,[ 0.8754795 ,  0.95285289,  1.03033239,  1.10803792,  1.18606854,
1.26443234,  1.34321265,  1.42258227,  1.5027733 ,  1.58394785,
1.66625727,  1.74998243,  1.8353173 ,  1.92227431,  2.01076693,
2.10064087,  2.19164551,  2.28346334,  2.37573696,  2.46825194,
2.56087699])

np.allclose(y99,[ 0.95227461,  0.97317995,  1.01159244,  1.08726908,  1.21260587,
1.31545644,  1.37829108,  1.42719649,  1.51308685,  1.59669367,
1.61486217,  1.64481078,  1.72970022,  1.83208819,  1.93312796,
2.05164767,  2.19326122,  2.34608425,  2.45023567,  2.5357288 ,
2.6357401 ])

plt.figure(0)
h=plt.plot(x,y, x,pp99(x),'g', x,pp9(x),'k', x,np.exp(x),'r')
plt.show()

def spline_alg_fit(x, y, t, c = 0):
    mu = 2 * (1-t)/(3*t)
    h = np.diff(x)
    p = 2 * (h[:-1] + h[1:])
    r = 3 / h
    f = -(r[:-1] + r[1:])
    R = np.diag(p, k=0) + np.diag(h[1:-1], k=1) + np.diag(h[1:-1], k=-1)
    Qt = np.zeros((f.size, f.size + 2), dtype=np.float)
    for i, v in enumerate((r[:-1], f, r[1:])):
        np.fill_diagonal(Qt[:,i:], v)
    Q = np.transpose(Qt)
    A = mu * np.matmul(Qt, Q) + R
    cur = c * np.ones(f.size, dtype=np.float)
    T = 1/3 * (1 - t) * (np.matmul(Q, cur) + np.matmul(cur, Qt))
    B = np.matmul(Qt, y) + np.matmul(Qt, T / t)
    b = np.linalg.solve(A, B)
    d = y - mu * np.matmul(Q, b) + T / t
    return d

def spline_numeric_fit(fit_y, x, y, t, c = 0):
    dx = np.diff(x)
    dydx = np.diff(fit_y) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = np.array([c, ddydx, c]) - c
    res = fit_y - y
    sumSpline = t * np.dot(res, res) + (1 - t) * np.dot(derivs, derivs) + res.size/2 * np.log(2*np.pi*1/(2*t)) + len(derivs)/2 * np.log(2*np.pi*1/(2*(1-t)))
    return sumSpline  
    
def spline_numeric_fit_vars(fit_y, x, y, tj, ti, c = 0):
    dx = np.diff(x)
    dydx = np.diff(fit_y) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = np.array([c, ddydx, c]) - c
    res = fit_y - y
    sumSpline = tj * np.dot(res, res) + ti * np.dot(derivs, derivs) + res.size/2 * np.log(2*np.pi*1/(2*tj)) + len(derivs)/2 * np.log(2*np.pi*1/(2*ti))
    return sumSpline      
    
def spline_numeric_fit_prob(fit, x, y, t, c = 0):
    dx = np.diff(x)
    dydx = np.diff(fit) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = [c, ddydx, c] - c
    res = fit - y
    term1 = -np.sum(np.log(sp.stats.norm.pdf(res, 0, np.sqrt(1/(2*t)))))
    term2 = -np.sum(np.log(sp.stats.norm.pdf(derivs, 0, np.sqrt(1/(2*(1-t))))))
    if not(np.isfinite(term1) & np.isfinite(term2)):
        print('error: non-finite value')
    return (term1 + term2)
    
def spline_numeric_fit_prob_vars(fit, x, y, v1, v2, c = 0):
    dx = np.diff(x)
    dydx = np.diff(fit) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = [c, ddydx, c] - c
    res = fit - y
    term1 = -np.sum(np.log(sp.stats.norm.pdf(res, 0, v1)))
    term2 = -np.sum(np.log(sp.stats.norm.pdf(derivs, 0, v2)))
    if not(np.isfinite(term1) & np.isfinite(term2)):
        print('error: non-finite value')
    return (term1 + term2)    

#class bounds_fit(object):
#     def __init__(self, xmax = [1.0], xmin = [0.0] ):
#         self.xmax = np.array(xmax)
#         self.xmin = np.array(xmin)
def Bcall(**kwargs):
    x = kwargs["x_new"]
    tmax = bool(np.all(x > 0))
    tmin = bool(np.all(x < 1))
    return tmax and tmin    
    
def spline_numeric_fit_prob_opt_t(par, x, y, epsilon, c = 0):
    global fit
    t = 1/(1 + epsilon * 10**par)
        
    def negnormP(fit):
        dx = np.diff(x)
        dydx = np.diff(fit) / dx
        ddydx = np.diff(dydx) / dx[:-1]
        derivs = [c, ddydx, c] - c
        res = fit - y
        retval = t * np.dot(res, res) + (1 - t) * np.dot(derivs, derivs) + res.size/2 * np.log(2*np.pi*1/(2*t)) + len(derivs)/2 * np.log(2*np.pi*1/(2*(1-t)))
        if (retval == -np.inf):
            retval = np.inf
        return retval
    #opt = sp.optimize.fmin(func = negnormP, x0 = s, full_output = 1)
    minimizer_kwargs_par = {"method": "BFGS"}
    opt = sp.optimize.basinhopping(negnormP, x0 = fit, minimizer_kwargs = minimizer_kwargs_par, niter = 3)
    fit = opt.x

#    if (opt.fun == np.inf):
#        val = opt.fun
#    else:
#        val = opt.fun
    val = -opt.fun
    return val   
    
def spline_numeric_fit_prob_opt_pars(pars, x, y, epsilon, c = 0):
    global fit
    tj = 1/(1 + epsilon * 10**(pars[0]))
    ti = 1 - tj
    vj = np.sqrt(1/(2*tj))
    vi = np.sqrt(1/(2*ti))
    
    if (len(pars) > 1):
        wk = np.exp(pars[1])
        norm_ent = sp.stats.norm.entropy(0, vj)
        ent_match = lambda vjq: np.abs(ratioProb.entropy(0, vjq, wk) - norm_ent)
        ent_opt_j = sp.optimize.fmin(func = ent_match, x0 = vj)
    
    def negnormP(fit):
        dx = np.diff(x)
        dydx = np.diff(fit) / dx
        ddydx = np.diff(dydx) / dx[:-1]
        derivs = [c, ddydx, c] - c
        res = fit - y
        if (len(pars) == 1):
            t1 = -np.sum(np.log(sp.stats.norm.pdf(res, 0, vj)))
            t2 = -np.sum(np.log(sp.stats.norm.pdf(derivs, 0, vi)))
        elif(len(pars) == 2):
            t1 = -np.sum(np.log(ratioProb.pdf(res, 0, ent_opt_j, wk)))
            t2 = -np.sum(np.log(sp.stats.norm.pdf(derivs, 0, vi))) 
        retval = t1 + t2
        if (retval == -np.inf):
            retval = np.inf    
        return retval
    #opt = sp.optimize.fmin(func = negnormP, x0 = s, full_output = 1)
    minimizer_kwargs_par = {"method": "BFGS"}
    opt = sp.optimize.basinhopping(negnormP, x0 = fit, minimizer_kwargs = minimizer_kwargs_par, niter = 5)
    fit = opt.x

    val = opt.fun
    return val
    
x = np.linspace(0, 5, 20)
ytrue = x*x*x
y = ytrue + 8*np.random.normal(0, 1, 20)
(a, b, r, tt, stderr) = sp.stats.linregress(x, y)
line_y = b + a * x
   
epsilonN = ((x[-1] - x[0])/(x.size - 1))**3/16
t = [] 
opt = []
totprob = []
sumSpline1 = []; sumSpline2 = []

for j in range(20): # do as per matlab
    t.append(1/(1 + epsilonN * 10**(j - 10)))
    #spline_opt = sp.optimize.fmin(func = spline_numeric_fit, x0 = y, args=(x, y, t), full_output = 1, maxiter = 10000)
    #spline_opt = sp.optimize.minimize(spline_numeric_fit, x0 = y, args=(x, y, t), method='Powell', options= {'maxiter': 100000})
    minimizer_kwargs = {"method": "BFGS", "args": (x, y, t[-1])}
    spline_opt = sp.optimize.basinhopping(spline_numeric_fit, x0 = y, minimizer_kwargs = minimizer_kwargs, niter = 20)                    
    #spline_opt2 = sp.optimize.basinhopping(spline_numeric_fit_prob, x0 = spline_opt.x, minimizer_kwargs = minimizer_kwargs, niter = 20)
    dx = np.diff(x)
    dydx = np.diff(spline_opt.x) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = [0, ddydx, 0]
    res = y - spline_opt.x
#    negnormP = lambda q: -np.sum(np.log(sp.stats.norm.pdf(res, 0, q)) + 
#                                         np.log(sp.stats.norm.pdf(derivs, 0, q*np.sqrt(t[-1]/(1 - t[-1])))))
#    opt.append(sp.optimize.fmin(func = negnormP, x0 = np.std(res)*2, full_output = 1)) #arbitrary initialization
    opt.append(spline_opt.fun)
#    resstd = np.sqrt(np.dot(res, res)/res.size)
#    derstd = np.sqrt(np.dot(derivs, derivs)/len(derivs))
#    totprob.append(np.sum(np.log(sp.stats.norm.pdf(res, 0, resstd))) + np.sum(np.log(sp.stats.norm.pdf(derivs, 0, derstd))))
#    
#    spline_opt3_B = intp.SmoothSpline(x, y, t[-1])
#    spline_opt3 = spline_opt3_B(x)
#    dydx = np.diff(spline_opt3) / dx
#    ddydx = np.diff(dydx) / dx[:-1]
#    derivs = [0, ddydx, 0]
#    res = y - spline_opt3
#    sumSpline1.append(t[-1] * np.dot(res, res) + (1 - t[-1]) * np.dot(derivs, derivs) + res.size/2 * np.log(2*np.pi*1/(2*t[-1])) + 
#                      len(derivs)/2 * np.log(2*np.pi*1/(2*(1-t[-1]))))
#    sumSpline2.append(t[-1] * np.dot(res, res) + (1 - t[-1]) * np.dot(derivs, derivs))
    
#prob = [p[1] for p in opt]    
    

fit = y
optval = sp.optimize.fmin(func = spline_numeric_fit_prob_opt_t, x0 = 5, args = (x, y, epsilonN), 
                          full_output = 1, maxiter = 10)    

plt.figure(1)    
testP5 = intp.SmoothSpline(x, y, p = .5)
algeT = spline_alg_fit(x, y, .5)
algeT2 = spline_alg_fit(x, y, .5, 2)
!!!!out = spline_opt.x
h = plt.plot(x, y, x, out, x, algeT, x, testP5(x))

spline_numeric_fit(algeT, x, y, t)
spline_numeric_fit(testP5(x), x, y, t)
spline_numeric_fit(spline_opt.x, x, y, t)
spline_numeric_fit(spline_opt2.x, x, y, t)
    
#v put in basin hopping, try as alternative to stepping through t       
    
# ratio dist
class ratioDist(sp.stats.rv_continuous):
    "Ratio distribution"
    def _argcheck(self, mu, s1, s2):
        return (s1 > 0) and (s2 > 0)
    
    def _pdf(self, x, mu, s1, s2):
        
        rt2 = np.sqrt(2)
        psum = s1 * s1 + x * x * s2 * s2
        prod = s1 * s2
        pmusum = s1 * s1 + x * mu * s2 * s2
        
        t1 = 1/(rt2 * np.pi * (psum) ** (3/2)) # problem line!!   OverflowError: (34, 'Result too large')
        t2 = np.exp(-(mu * mu/(2 * s1 * s1) + 1/(2 * s2 * s2)))
        t3 = rt2 * prod * np.sqrt(psum)
        t4 = sp.special.erf(pmusum/(rt2 * prod * np.sqrt(psum)))
        t24 = np.exp(-(mu * mu/(2 * s1 * s1) + 1/(2 * s2 * s2)) + pmusum * pmusum/(2 * prod * prod * psum))
        
        return t1 * (t2 * t3 + t24 * np.sqrt(np.pi) * t4 * pmusum)    
    
    
def spline_numeric_fit_prob_vars_ratio(fit, x, y, a, b, c = 0, d = 0, cu = 0):
    dx = np.diff(x)
    dydx = np.diff(fit) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = [cu, ddydx, cu] - cu
    res = fit - y
    if (c != 0):
        term1 = -np.sum(np.log(ratioProb.pdf(res, 0, a, c)))
    else:
        term1 = -np.sum(np.log(sp.stats.norm.pdf(res, 0, a)))
    if (d != 0):
        term2 = -np.sum(np.log(ratioProb.pdf(derivs, 0, b, d)))
    else:
        term2 = -np.sum(np.log(sp.stats.norm.pdf(derivs, 0, b)))
    #if not(np.isfinite(term1) & np.isfinite(term2)):
    #    print('error: non-finite value')
    return (term1 + term2)    
    
def spline_numeric_fit_OU(fit_y, x, y, t, c = 0, k = 1):
    fit_y = np.array(fit_y)
    y = np.array(y)
    dx = np.diff(x[1:])
    dydx = np.diff(fit_y) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = np.array([c, *list(ddydx), c]) - c
    yprev = y[:-1]
    res = yprev - y[1:] - k * (yprev - fit_y)
    sumSpline = t * np.dot(res, res) + (1 - t) * np.dot(derivs, derivs) + res.size/2 * np.log(2*np.pi*1/(2*t)) + len(derivs)/2 * np.log(2*np.pi*1/(2*(1-t)))
    return sumSpline  
    
def spline_numeric_fit_vars_OU(fit_y, x, y, tj, ti, c = 0, k = 1):
    fit_y = np.array(fit_y)
    y = np.array(y)
    dx = np.diff(x[1:])
    dydx = np.diff(fit_y) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = np.array([c, *list(ddydx), c]) - c
    yprev = y[:-1]
    res = y[1:] - yprev + k * (yprev - fit_y)
    sumSpline = tj * np.dot(res, res) + ti * np.dot(derivs, derivs) # + res.size/2 * np.log(2*np.pi*1/(2*tj)) + len(derivs)/2 * np.log(2*np.pi*1/(2*ti))
    return sumSpline      
    
def spline_numeric_fit_prob_OU(fit_y, x, y, t, c = 0, k = 1):
    fit_y = np.array(fit_y)
    y = np.array(y)
    dx = np.diff(x[1:])
    dydx = np.diff(fit_y) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = [c, ddydx, c] - c
    yprev = y[:-1]
    res = yprev - y[1:] - k * (yprev - fit_y)
    term1 = -np.sum(np.log(sp.stats.norm.pdf(res, 0, np.sqrt(1/(2*t)))))
    term2 = -np.sum(np.log(sp.stats.norm.pdf(derivs, 0, np.sqrt(1/(2*(1-t))))))
    if not(np.isfinite(term1) & np.isfinite(term2)):
        print('error: non-finite value')
    return (term1 + term2)    
    
    
ratioProb = ratioDist(name = 'ratio')
size = 200
ratio_x = np.linspace(0, 5, size)
rands = np.array([])
while (rands.size < size):
    newrs = ratioProb.rvs(mu = 0, s1 = 1.5, s2 = 0.4, size = 50)
    rands = np.append(rands, newrs) 

truey2 = ratio_x*ratio_x*ratio_x
ratio_y = truey2 + rands
(a , b , r, tt, stderr) = sp.stats.linregress(ratio_x, ratio_y)    
epsilon = ((ratio_x[-1] - ratio_x[0])/(ratio_x.size - 1))**3/16
#
t = []; prob = []; stdres = []; stdderiv = []; pvalue = []

for j in range(20):
    t.append(1/(1 + epsilon * 10**(2*j - 6)))
    #algeT = spline_alg_fit(ratio_x, ratio_y, 1 - t[-1])
    minimizer_kwargs = {"method": "BFGS", "args": (ratio_x, ratio_y, t[-1])}
    spline_opt_0 = sp.optimize.basinhopping(spline_numeric_fit, x0 = ratio_y, minimizer_kwargs = minimizer_kwargs, niter = 10)                                                                   
    spline_opt = sp.optimize.basinhopping(spline_numeric_fit_prob, x0 = spline_opt_0.x, minimizer_kwargs = minimizer_kwargs, niter = 10)
    pvalue.append(spline_opt_0.fun)
    dx = np.diff(ratio_x)
    dydx = np.diff(spline_opt.x) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = [0, ddydx, 0]
    res = ratio_y - spline_opt.x
    negnormP = lambda q: -np.sum(np.log(sp.stats.norm.pdf(res, 0, q*np.sqrt(1/(2*t[-1])))) + 
                                         np.log(sp.stats.norm.pdf(derivs, 0, q*np.sqrt(1/(2*(1-t[-1]))))))
    opt = sp.optimize.fmin(func = negnormP, x0 = np.std(res)/np.sqrt(1/(2*t[-1])), full_output = 1)
    if not(np.isfinite(opt[1])):
        opt = sp.optimize.fmin(func = negnormP, x0 = np.std(derivs)/np.sqrt(1/(2*(1-t[-1]))), full_output = 1)
    stdres.append(opt[0] * np.sqrt(1/(2*t[-1])))
    stdderiv.append(opt[0] * np.sqrt(1/(2*(1-t[-1]))))
    prob.append(opt[1])


size = 9
size1 = 10
vi = []; vj = [];
prob = np.zeros((size1, size)); stdres = np.zeros((size1, size)); stdderiv = np.zeros((size1, size))
istep = np.logspace(-8, 8, num = size)
    
for j in range(size1):
    tj = 1/(1 + epsilon * 10**(2*j+3))    #(j + 3)) #13
    vj.append(np.sqrt(1/(2*tj)))
    for i in range(size):
        ti = istep[i]  # 1 - 1/(1 + epsilon * 10**(2*i + 3)) #13
        vi.append(np.sqrt(1/(2*ti)))
        minimizer_kwargs_0 = {"method": "BFGS", "args": (ratio_x, ratio_y, tj, ti)}
        # next line could init with previous loop spline instead of ratio_y 
        spline_opt_0 = sp.optimize.basinhopping(spline_numeric_fit_vars, x0 = ratio_y, minimizer_kwargs = minimizer_kwargs_0, niter = 5)                                
        minimizer_kwargs = {"method": "BFGS", "args": (ratio_x, ratio_y, vj[-1], vi[-1])}
        spline_opt = sp.optimize.basinhopping(spline_numeric_fit_prob_vars, x0 = spline_opt_0.x, minimizer_kwargs = minimizer_kwargs, niter = 20)
        dx = np.diff(ratio_x)
        dydx = np.diff(spline_opt.x) / dx
        ddydx = np.diff(dydx) / dx[:-1]
        derivs = [0, ddydx, 0]
        res = ratio_y - spline_opt.x
        negnormP = lambda q: -np.sum(np.log(sp.stats.norm.pdf(res, 0, q*vj[-1])) + 
                                         np.log(sp.stats.norm.pdf(derivs, 0, q*vi[-1])))
        opt = sp.optimize.fmin(func = negnormP, x0 = np.std(res)/vj[-1], full_output = 1)
        if not(np.isfinite(opt[1])):
            opt = sp.optimize.fmin(func = negnormP, x0 = np.std(derivs)/vi[-1], full_output = 1)
        stdres[j, i] = opt[0] * vj[-1]
        stdderiv[j, i] = opt[0] * vi[-1]
        prob[j, i] = opt[1]


plt.plot([prob[k, k] for k in range(size1)])
plt.plot([stdres[k, k] for k in range(size)])
plt.plot([stdderiv[k, k] for k in range(size)])
im = plt.imshow(prob, cmap='hot', origin='lower' ) 
plt.colorbar(im, orientation='horizontal')  
plt.show()

minval = np.amin(prob)
minp = np.unravel_index(prob.argmin(), prob.shape)


# ratio spline

def calcProb(xs, ys, spline, vj, vi, wk = 0, wl = 0, c = 0):       
    dx = np.diff(xs)
    dydx = np.diff(spline) / dx
    ddydx = np.diff(dydx) / dx[:-1]
    derivs = np.array([c, ddydx, c]) - c
    res = ys - spline
    if (wk != 0):
        if (wl != 0):
            negnormP = lambda q: -np.sum(np.log(ratioProb.pdf(res, 0, q*vj, wk)) + 
                                         np.log(ratioProb.pdf(derivs, 0, q*vi, wl)))
        else:
            negnormP = lambda q: -np.sum(np.log(ratioProb.pdf(res, 0, q*vj, wk)) + 
                                         np.log(sp.stats.norm.pdf(derivs, 0, q*vi)))
    else:
        negnormP = lambda q: -np.sum(np.log(sp.stats.norm.pdf(res, 0, q*vj)) + 
                                         np.log(sp.stats.norm.pdf(derivs, 0, q*vi)))
    opt = sp.optimize.fmin(func = negnormP, x0 = np.std(res)/vj, full_output = 1)
    if not(np.isfinite(opt[1])):
        opt = sp.optimize.fmin(func = negnormP, x0 = np.std(derivs)/vi, full_output = 1)
    return opt

sampleSz = 25
epsilon = ((ratio_x[sampleSz - 1] - ratio_x[0])/(sampleSz - 1))**3/16
size0 = 1 #5
size1 = 20 #10
size2 = 0 #5
vi = []; vj = []; wk = []; wl = []
prob0 = np.zeros((size0)); stdres1 = np.zeros((size0)); stdderiv1 = np.zeros((size0))
prob1 = np.zeros((size0, size1)); stdres2 = np.zeros((size0, size1)); stdderiv2 = np.zeros((size0, size1))
prob2 = np.zeros((size0, size1, size2)); stdres3 = np.zeros((size0, size1, size2)); stdderiv3 = np.zeros((size0, size1, size2))
istep = np.logspace(8, -8, num = size2)
xs = ratio_x[0:sampleSz]
ys = ratio_y[0:sampleSz]
pvals = []
    
for j in range(size0):
    tj = 1/(1 + epsilon * 10**(1))  # 1/(1 + epsilon * 10**(2*j - 5))
    vj.append(np.sqrt(1/(2*tj)))
    ti = 1 - tj
    vi.append(np.sqrt(1/(2*ti)))
    minimizer_kwargs_0 = {"method": "BFGS", "args": (xs, ys, tj, ti)}
    # next line could init with previous loop spline instead of ratio_y 
    spline_opt_0 = sp.optimize.basinhopping(spline_numeric_fit_vars, x0 = ys, minimizer_kwargs = minimizer_kwargs_0, niter = 10)                                
    pvals.append(spline_opt_0.fun)
    
    if (size1 > 0):
        for k in range(size1):                    
#            uk =  1/(1 + epsilon * 10**(6*k - 20))
#            wk.append(np.sqrt(1/(2*uk)))
            wk.append(np.exp((k-5)))
            minimizer_kwargs = {"method": "BFGS", "args": (xs, ys, vj[-1], vi[-1], wk[-1])}
            if (k == 0):
                x0 = spline_opt_0.x
            else:
                x0 = spline_opt_1.x
            spline_opt_1 = sp.optimize.basinhopping(spline_numeric_fit_prob_vars_ratio, x0 = x0, minimizer_kwargs = minimizer_kwargs, niter = 10)
            
            if (size2 > 0):
                for l in range(size2):                    
                    ul =  istep[l]
                    wl.append(np.sqrt(1/(2*ul)))
                    minimizer_kwargs = {"method": "BFGS", "args": (xs, ys, vj[-1], vi[-1], wk[-1], wl[-1])}
                    if (l == 0):
                        x0 = spline_opt_1.x
                    else:
                        x0 = spline_opt_2.x    
                    spline_opt_2 = sp.optimize.basinhopping(spline_numeric_fit_prob_vars_ratio, x0 = x0, minimizer_kwargs = minimizer_kwargs, niter = 10)
                    opt = calcProb(xs, ys, spline_opt_2.x, vj[-1], vi[-1], wk[-1], wl[-1])
                    stdres3[j, k, l] = opt[0] * vj[-1]
                    stdderiv3[j, k, l] = opt[0] * vi[-1]
                    prob2[j, k, l] = opt[1]
                    print('finished ' + str(j) + ', ' + str(k) + ', ' + str(l))
            else:
                opt = calcProb(xs, ys, spline_opt_1.x, vj[-1], vi[-1], wk[-1])
                stdres2[j, k] = opt[0] * vj[-1]
                stdderiv2[j, k] = opt[0] * vi[-1]
                prob1[j, k] = opt[1]
                print('finished ' + str(j) + ', ' + str(k))    
    else:
        opt = calcProb(xs, ys, spline_opt_0.x, vj[-1], vi[-1])
        stdres1[j] = opt[0] * vj[-1]
        stdderiv1[j] = opt[0] * vi[-1]
        prob0[j] = opt[1]
        print('finished ' + str(j))        
            

plt.plot([prob1[k, k] for k in range(size0)])
plt.plot([stdres2[k, k] for k in range(size1)])
plt.plot([stdderiv2[k, k] for k in range(size1)])
im = plt.imshow(prob1, cmap = 'hot', origin = 'lower') 
plt.colorbar(im, orientation='horizontal')  
plt.show()

maxval = np.amax(prob1)
maxp = np.unravel_index(prob1.argmax(), prob1.shape)
minval = np.amin(prob1)
minp = np.unravel_index(prob1.argmin(), prob1.shape)


# line data
size = 50
ratio_x = np.linspace(0, 10, size)
rands = np.array([])
while (rands.size < size):
    try:
        newrs = ratioProb.rvs(mu = 0, s1 = 1.5, s2 = 0.001, size = 50)
        rands = np.append(rands, newrs) 
    except:
        continue
ratio_ln_y = ratio_x + rands
(a , b , r, tt, stderr) = sp.stats.linregress(ratio_x, ratio_ln_y)
plt.figure(1)      
h1 = plt.plot(ratio_x, ratio_ln_y)
predict_y = b + a * ratio_x
h2 = plt.plot(ratio_x, predict_y)
devs = [np.sqrt(np.sum((predict_y - ratio_ln_y) * (predict_y - ratio_ln_y))/ratio_ln_y.size), .05]

        
sampleSz = 50
epsilon = ((ratio_x[sampleSz - 1] - ratio_x[0])/(sampleSz - 1))**3/16
size0 = 10
size1 = 0
size2 = 0
vi = []; vj = []; wk = []; wl = []
prob0 = np.zeros((size0)); stdres1 = np.zeros((size0)); stdderiv1 = np.zeros((size0))
prob1 = np.zeros((size0, size1)); stdres2 = np.zeros((size0, size1)); stdderiv2 = np.zeros((size0, size1))
prob2 = np.zeros((size0, size1, size2)); stdres3 = np.zeros((size0, size1, size2)); stdderiv3 = np.zeros((size0, size1, size2))
istep = np.logspace(4, -4, num = size2)
xs = ratio_x[0:sampleSz]
ys = ratio_ln_y[0:sampleSz]
sample = rands[0:sampleSz]
pvals = []
pvals2 = np.zeros((size0, size1))
pvals3 = np.zeros((size0, size1, size2))
    
for j in range(size0):
    tj = 1/(1 + epsilon * 10**(2*j - 5))  #5
    vj.append(np.sqrt(1/(2*tj)))
    ti = 1 - tj
    vi.append(np.sqrt(1/(2*ti)))
    minimizer_kwargs_0 = {"method": "BFGS", "args": (xs, ys, tj)}
    # next line could init with previous loop spline instead of ratio_y 
    spline_opt_0 = sp.optimize.basinhopping(spline_numeric_fit, x0 = ys, minimizer_kwargs = minimizer_kwargs_0, niter = 10)                                
    pvals.append(spline_opt_0.fun)
#    minimizer_kwargs = {"method": "BFGS", "args": (xs, ys, vj[-1], vi[-1], 0.8)}
#    spline_opt_1 = sp.optimize.basinhopping(spline_numeric_fit_prob_vars_ratio, x0 = spline_opt_0.x, minimizer_kwargs = minimizer_kwargs, niter = 20)

    if (size1 > 0):
        for k in range(size1):                    
            wk.append(0.5)  #(0.0001 + 0.2*k)   #(np.exp(2*k-5))
            # adjust vj[-1] to maintain some variance-like property at constant
            norm_ent = sp.stats.norm.entropy(0, vj[-1])
            ent_match = lambda vjq: np.abs(ratioProb.entropy(0, vjq, wk[-1]) - norm_ent)
            ent_opt_j = sp.optimize.fmin(func = ent_match, x0 = vj[-1])
            kwargs_1 = {"method": "BFGS", "args": (xs, ys, ent_opt_j, vi[-1], wk[-1])}
            if (k == 0):
                x0 = spline_opt_0.x
            else:
                x0 = spline_opt_1.x
            spline_opt_1 = sp.optimize.basinhopping(spline_numeric_fit_prob_vars_ratio, x0 = x0, minimizer_kwargs = kwargs_1, niter = 10)
            pvals2[j, k] = spline_opt_1.fun
 
            if (size2 > 0):
                for l in range(size2):                    
                    ul =  istep[l]
                    wl.append(np.sqrt(1/(2*ul)))
                    norm_ent = sp.stats.norm.entropy(0, vi[-1])
                    ent_match = lambda viq: np.abs(ratioProb.entropy(0, viq, wl[-1]) - norm_ent)
                    ent_opt_i = sp.optimize.fmin(func = ent_match, x0 = vi[-1], full_output = 1)
                    kwargs_2 = {"method": "BFGS", "args": (xs, ys, ent_opt_j, ent_opt_i, wk[-1], wl[-1])}
                    if (l == 0):
                        x0 = spline_opt_1.x
                    else:
                        x0 = spline_opt_2.x    
                    spline_opt_2 = sp.optimize.basinhopping(spline_numeric_fit_prob_vars_ratio, x0 = x0, minimizer_kwargs = kwargs_2, niter = 5)
                    pvals3[j, k, l] = spline_opt_2.fun
#                    opt = calcProb(xs, ys, spline_opt_2.x, vj[-1], vi[-1], wk[-1], wl[-1])
#                    stdres3[j, k, l] = opt[0] * ent_opt_j
#                    stdderiv3[j, k, l] = opt[0] * ent_opt_i
#                    prob2[j, k, l] = opt[1]
                    print('finished ' + str(j) + ', ' + str(k) + ', ' + str(l))
            else:
                opt = calcProb(xs, ys, spline_opt_1.x, ent_opt_j, vi[-1], wk[-1])
                stdres2[j, k] = opt[0] * ent_opt_j
                stdderiv2[j, k] = opt[0] * vi[-1]
                prob1[j, k] = opt[1]
                print('finished ' + str(j) + ', ' + str(k))    
    else:
        opt = calcProb(xs, ys, spline_opt_0.x, vj[-1], vi[-1])
        stdres1[j] = opt[0] * vj[-1]
        stdderiv1[j] = opt[0] * vi[-1]
        prob0[j] = opt[1]
        print('finished ' + str(j))   
        
np.unravel_index(pvals3.argmin(), pvals3.shape)
im = plt.imshow(pvals2, cmap = 'hot', origin = 'lower')        
# try maximising prob for res and derivs separately

(a , b , r, tt, stderr) = sp.stats.linregress(xs, ys)
plt.figure(1)      
#h1 = plt.plot(xs, ys)
predict_y = b + a * xs
h2 = plt.plot(xs, predict_y)

fit = ys        
optim0 = sp.optimize.fmin(func = spline_numeric_fit_prob_opt_t, x0 = 5, args = (xs, ys, epsilon), 
                          full_output = 1, maxiter = 5)
# must begin this with already smoothed fit! and its slow
optim1 = sp.optimize.fmin(func = spline_numeric_fit_prob_opt_pars, x0 = [optim0[0], 0], args = (xs, ys, epsilon), 
                          full_output = 1)           

# with curvature
size0 = 7
curvsize = 4
vi = []; vj = []; 
curv = np.linspace(0, 30, curvsize)
pvals = []
pvals2 = np.zeros((size0, curvsize))
epsilonN = ((x[-1] - x[0])/(x.size - 1))**3/16
xs = x[0:sampleSz]
ys = y[0:sampleSz]
        
for j in range(size0):
    tj = 1/(1 + epsilonN * 10**(2*j - 2))
    vj.append(np.sqrt(1/(2*tj)))
    ti = 1 - tj
    vi.append(np.sqrt(1/(2*ti)))
    minimizer_kwargs_0 = {"method": "BFGS", "args": (xs, ys, tj)}
    # next line could init with previous loop spline instead of ratio_y 
    spline_opt_0 = sp.optimize.basinhopping(spline_numeric_fit, x0 = ys, minimizer_kwargs = minimizer_kwargs_0, niter = 20)                                
    pvals.append(spline_opt_0.fun)
    print('finished ' + str(j))
    
    if (curvsize > 0):
        for k in range(curvsize):
            minimizer_kwargs_1 = {"method": "BFGS", "args": (xs, ys, tj, ti, curv[k])}
            if (k == 0):
                x0 = spline_opt_0.x
            else:
                x0 = spline_opt_1.x
            spline_opt_1 = sp.optimize.basinhopping(spline_numeric_fit_vars, x0 = x0, minimizer_kwargs = minimizer_kwargs_1, niter = 10)                                
            pvals2[j, k] = spline_opt_1.fun                         
            print('finished ' + str(j) + ', ' + str(k))  

if (curvsize > 0):
    im = plt.imshow(pvals2, cmap = 'hot', origin = 'lower')


# with Ornstein-Uhlenbeck process
ratioProb = ratioDist(name = 'ratio')
size = 200
ratio_x = np.linspace(4.8, 5.8, size)
rands = np.array([])
while (rands.size < size):
    try:
        newrs = ratioProb.rvs(mu = 0, s1 = 1.5, s2 = 0.001, size = 50)
        rands = np.append(rands, newrs) 
    except:
        continue

    
r = .05
ratio_ln_y_OU = []    
ratio_ln_y_OU.append(ratio_x[0] + r * rands[0])  
for i in range(size - 1):
    ratio_ln_y_OU.append(ratio_ln_y_OU[i] + rands[i+1] - r * (ratio_ln_y_OU[i] - ratio_x[i+1])) 
pd.DataFrame(ratio_ln_y_OU).to_csv('ratio_ln.csv')
    
realY = ratio_x*ratio_x*ratio_x / 50.0
ratio_cub_y_OU = []    
ratio_cub_y_OU.append(realY[0] + r * rands[0])  
for i in range(size - 1):
    ratio_cub_y_OU.append(ratio_cub_y_OU[i] + rands[i+1] - r * (ratio_cub_y_OU[i] - realY[i+1]))

k1 = -.22; k2 = 1.25;
y_test = np.zeros(sampleSz)
y_test[0] = rands[0]
y_test[1] = rands[1] + (1 - k1) * y_test[0]
for i in range(sampleSz - 1):
    y_test[i+1] = rands[i+1] + (1 - k1) * y_test[i] + (1 - k2) * y_test[i-1] + 500*(ratio_x[i+1] - 4.8) * (k1 + k2 - 1)
pd.DataFrame(y_test).to_csv('y_test2.csv')    
    
sampleSz = 100
epsilon = ((ratio_x[sampleSz - 1] - ratio_x[0])/(sampleSz - 1))**3/16
size0 = 20
OUsize = 7
vi = []; vj = []; 
OUvals = np.linspace(0, .25, OUsize)
pvals = []
pvals2 = np.zeros((OUsize, size0))
xs = ratio_x[0:sampleSz]
ys = list(50*(dfSPY['logMedian'][0:sampleSz]))
t = 1/(1 + epsilon * 10**(4)) # estimate big for line
k = 1 #6
j = 11
        
for k in range(OUsize):
    args_1 = {"method": "BFGS", "args": (xs, ys, t, 1-t, 0, OUvals[k])}
    if (k == 0):
        x0 = ys[1:]
    else:
        x0 = spline_opt_1.x
    spline_opt_1 = sp.optimize.basinhopping(spline_numeric_fit_vars_OU, x0 = x0, 
                    minimizer_kwargs = args_1, niter = 20)                                
    pvals.append(spline_opt_1.fun)                       
    print('finished ' + str(k))  

    if (size0 > 0):
        for j in range(size0):
            tj = 1/(1 + epsilon * 10**(2*j - 20))
            vj.append(np.sqrt(1/(2*tj)))
            ti = 1 - tj
            vi.append(np.sqrt(1/(2*ti)))
            args_0 = {"method": "BFGS", "args": (xs, ys, tj, ti, 0, OUvals[k])} 
            # next line could init with previous loop spline instead of ratio_y 
            if (j == 0):
                x0 = ys[1:]
            else:
                x0 = spline_opt_0.x
            spline_opt_0 = sp.optimize.basinhopping(spline_numeric_fit_vars_OU, x0 = x0, 
                            minimizer_kwargs = args_0)   # niter = 20,                              
            pvals2[k, j] = spline_opt_0.fun
            print('finished ' + str(k) + ', ' + str(j))  
    
    
if (size0 > 0):
    np.unravel_index(pvals2.argmin(), pvals2.shape)
    im = plt.imshow(pvals2, cmap = 'hot', origin = 'lower')    
    

# test 2nd deriv distribution for some curve
dx = np.diff(ratio_x)
dydx = np.diff(realY) / dx
ddydx = np.diff(dydx) / dx[:-1]
derivs = np.array(ddydx)
xmin = [0.0001, 0.0001]
xmax = [10, 10]
bounds = [(low, high) for low, high in zip(xmin, xmax)]
min_kwargs_test = {"method": "L-BFGS-B", "bounds": bounds}
calcRatPars = lambda q: -np.sum(np.log(ratioProb.pdf(derivs, 0, q[0], q[1])))
par_test = sp.optimize.basinhopping(func = calcRatPars, x0 = [1., 0.1], minimizer_kwargs = min_kwargs_test)

# analyse 2nd deriv mean, res for fitted spline
dx = np.diff(xs[1:])  # ratio_x[1:]
dydx = np.diff(spline_opt_0.x) / dx  # ys
ddydx = np.diff(dydx) / dx[-1]
derivs = np.array(ddydx)
curvMean = np.mean(derivs)
ya = np.array(ys)
yprev = ya[:-1]
res = ya[1:] - yprev + OUvals[k] * (yprev - spline_opt_0.x)
tj * np.dot(res, res) + ti * np.dot(derivs, derivs)

# calculate k for fitted spline
ydata = ratio_cub_y_OU
x0 = [1., 0.01]

def findK(k, ydata, spline):
    global x0
    ydata = np.array(ydata)
    spline = np.array(spline)
    yprev = ydata[:-1]
    res = ydata[1:] - yprev + k * (yprev - spline)
    resRatPars = lambda q: -np.sum(np.log(ratioProb.pdf(res, 0, q[0], q[1])))
    par_test = sp.optimize.basinhopping(func = resRatPars, x0 = x0, minimizer_kwargs = min_kwargs_test)
    x0 = par_test.x
    return par_test.fun
    
optimK = sp.optimize.fmin(func = findK, x0 = OUvals[k], args = (ydata, spline_opt_1.x))  

# check that calculated curvature and k are close to set values, tests model validity

# for market data, find t and OU pars


