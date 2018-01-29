# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:25:05 2017

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas_datareader.data as web
import pandas as pd
import datetime
import time
import numpy.polynomial.polynomial as poly
from sklearn import linear_model
from sklearn.linear_model import TheilSenRegressor
import itertools
from statsmodels.api import robust
import operator
import os
import pickle


os.chdir('C:\\Dropbox\\trading')


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

class MyTakeStep(object):
    def __init__(self, stepsize = 1.5):
        self.stepsize = stepsize
    def __call__(self, x):
        s = self.stepsize
        x[0] += np.random.uniform(-s, s)
        x[1:] += np.random.uniform(-4*s, 4*s, x[1:].shape)
        return x


ratioProb = ratioDist(name = 'ratio')


def init():
    
    dfRaw = pd.read_csv("C:\\Dropbox\\trading\\IB_hist_data\\NVDA_150d_15m.csv", header = None,)
    dfRaw['Median'] = dfRaw.apply(lambda row: np.median([np.median([row[2], 
                        row[5]]), np.median([row[3], row[4]])]), axis=1)
    
    #dfRaw['Group'] = [np.floor(x / 13) for x in range(dfRaw.shape[0])]
    dfRaw[1] = pd.to_datetime(dfRaw[1])
    dfRaw['Group'] = dfRaw[1].dt.date
    dfRaw["GCount"] = dfRaw.groupby(['Group']).cumcount()
    dfRaw['Group2'] = np.floor(dfRaw["GCount"] / 13)
    
    dfData = pd.DataFrame(dfRaw.groupby(['Group', 'Group2'])['Median'].median())
    dfData['logMedian'] = np.log(dfData['Median'])
    
    date_times = pd.DataFrame(dfRaw.groupby(['Group', 'Group2'])[1].first())
    #date_times = pd.DataFrame(pd.to_datetime(date_times1))
    
    for index, row in date_times.iterrows():
        if (index[1] == 0):
            row[1] = row[1].replace(hour = 9, minute = 30)
        else:
            row[1] = row[1].replace(hour = 12, minute = 45)
            
    dfData.index = date_times.ix[:,1]
    dfData.to_pickle('NVDA_2Daily_150')


def regress_numeric_line_AR_ent(line_pars, x, y, do_print, mu_l):
    global ARval
    global dev
    y = np.array(y)
    fit_y = line_pars[1] + line_pars[0] * x
    yprev = y[:-1]
    yprev2 = y[:-2]

    def findBestAR(pars, y, yprev, yprev2, fit_y, k2 = 1):
        k1 = pars[0]
        vs = pars[1:]
        res = y[2:] - (1 - k1) * yprev[1:] - (1 - k2) * yprev2 - fit_y[2:] * (k1 + k2 - 1)
        new_ent = -np.sum(np.log(ratioProb.pdf(res, vs[2], vs[0], vs[1])))
        return new_ent / res.size

    k1 = 0.4; k2 = 1
    res_init = y[2:] - (1 - k1) * yprev[1:] - (1 - k2) * yprev2 - fit_y[2:] * (k1 + k2 - 1)
    std_init = np.std(res_init)
    xmin = [.15, std_init / 20, std_init / 20, -mu_l * std_init]
    xmax = [1.3, 2.5 * std_init, 2.5 * std_init, mu_l * std_init]
    bounds = [(low, high) for low, high in zip(xmin, xmax)]
    x0_vals = [ARval, dev[0], dev[1], dev[2]]
    for i in range(len(x0_vals)):
        if (bounds[i][0] > x0_vals[i]):
            x0_vals[i] = bounds[i][0]
        if (bounds[i][1] < x0_vals[i]):
            x0_vals[i] = bounds[i][1]
#    k_opt = sp.optimize.minimize(findBestAR, x0 = ARval, method = 'L-BFGS-B', 
#                                bounds = bounds)
    k_kwargs = {"method": "L-BFGS-B", "args": (y, yprev, yprev2, fit_y), "bounds": bounds}
    k_opt = sp.optimize.basinhopping(findBestAR, x0 = x0_vals, minimizer_kwargs = k_kwargs,
                    niter = 5)
    
    ARval = k_opt.x[0]
    dev = k_opt.x[1:]
    if do_print:
        print(ARval, dev)
    return k_opt.fun

def par_limits(ARval, dev, std_init, mu_l):
    if ((ARval - 0.15) / 0.2 < 0.02):
        ARval_limit = True
    elif ((1.3 - ARval) / 1.3 < 0.02):
        ARval_limit = True
    else:
        ARval_limit = False
        
    if ((dev[0] - std_init / 20) / (std_init / 20) < 0.02):
        dev0 = True
    elif ((2.5 * std_init - dev[0]) / (2.5 * std_init) < 0.02):
        dev0 = True
    else:
        dev0 = False
        
    if ((dev[1] - std_init / 20) / (std_init / 20) < 0.02):
        dev1 = True
    elif ((2.5 * std_init - dev[1]) / (2.5 * std_init) < 0.02):
        dev1 = True
    else:
        dev1 = False
        
    if ((dev[2] + mu_l * std_init) / (-mu_l * std_init) < 0.02):
        dev2 = True
    elif ((mu_l * std_init - dev[2]) / (mu_l * std_init) < 0.02):
        dev2 = True
    else:
        dev2 = False
        
    hit_limits = (ARval_limit, dev0, dev1, dev2)
    return hit_limits
        
def AR_line_calc(x, y, mu_l):
    # random dist'n, AR value, and line estimates, eg. from Matlab 'detrend' & 'ar' 
    global dev 
    global ARval
    y = np.array(y)
    valid = ~np.isnan(y)
    y = y[valid]
    x = x[valid]
    if sum(np.isnan(y)) / y.size > 0.1:
        nan_valid = False
        print('error: too many nan values, returning')
        return
    else:
        nan_valid = True
        
    (a, b, r, tt, stderr) = sp.stats.linregress(x, y)
    init_fit = b + a * x
    res_ols = y - init_fit
    std_res_ols = np.std(res_ols)
    ols_val = -np.sum(np.log(sp.stats.norm.pdf(res_ols, np.mean(res_ols), std_res_ols))) \
        * (len(y) - 2) / len(y)
    print('initial OLS fun value (length normalized): ', ols_val)
    
    yprev = y[:-1]
    yprev2 = y[:-2] 
    k1 = 0.4; k2 = 1                   
    res_init = y[2:] - (1 - k1) * yprev[1:] - (1 - k2) * yprev2 - init_fit[2:] * (k1 + k2 - 1)
    std_init = np.std(res_init)
    ols_val_AR = -np.sum(np.log(sp.stats.norm.pdf(res_init, np.mean(res_init), std_init)))
    print('initial OLS with AR fun value: ', ols_val_AR)
    
    dev = [std_init / 2, std_init / 2, 0]
    ARval = 0.4
    print('initial ARval: ', ARval)
    print('initial dev: ', dev)
    mytakestep = MyTakeStep()
    if a > 0:
        xmin = [a * 0.75, b * 0.9]
        xmax = [a * 1.33, b * 1.1]
    else:
        xmin = [a * 1.33, b * 0.9]
        xmax = [a * 0.75, b * 1.1]
    print('a range: ', xmax[0], xmin[0])
    print('b range: ', xmax[1], xmin[1])
    bounds = [(low, high) for low, high in zip(xmin, xmax)]
    kwargs = {"method": "L-BFGS-B", "args": (x, y, False, mu_l), "bounds": bounds}
    par_opt = sp.optimize.basinhopping(regress_numeric_line_AR_ent, x0 = [a, b], minimizer_kwargs = kwargs,
                    niter = 5, take_step = mytakestep)
    
    hit_limits = par_limits(ARval, dev, std_init, mu_l)
    return [y, par_opt, hit_limits, nan_valid, ARval, dev]

def regress_numeric_line_AR_ent_final(line_pars, x, y, do_print, mu_l):
    y = np.array(y)
    fit_y = line_pars[1] + line_pars[0] * x
    yprev = y[:-1]
    yprev2 = y[:-2]

    def findBestAR(pars, y, yprev, yprev2, fit_y, k2 = 1):
        k1 = pars[0]
        vs = pars[1:]
        res = y[2:] - (1 - k1) * yprev[1:] - (1 - k2) * yprev2 - fit_y[2:] * (k1 + k2 - 1)
        new_ent = -np.sum(np.log(ratioProb.pdf(res, vs[2], vs[0], vs[1])))
        return new_ent / res.size

    k1 = 0.4; k2 = 1
    res_init = y[2:] - (1 - k1) * yprev[1:] - (1 - k2) * yprev2 - fit_y[2:] * (k1 + k2 - 1)
    std_init = np.std(res_init)
    xmin = [.15, std_init / 20, std_init / 20, -mu_l * std_init]
    xmax = [1.3, 2.5 * std_init, 2.5 * std_init, mu_l * std_init]
    bounds = [(low, high) for low, high in zip(xmin, xmax)]
    x0_vals = [0.4, std_init / 2, std_init / 2, 0]
    for i in range(len(x0_vals)):
        if (bounds[i][0] > x0_vals[i]):
            x0_vals[i] = bounds[i][0]
        if (bounds[i][1] < x0_vals[i]):
            x0_vals[i] = bounds[i][1]
#    k_opt = sp.optimize.minimize(findBestAR, x0 = ARval, method = 'L-BFGS-B', 
#                                bounds = bounds)
    k_kwargs = {"method": "L-BFGS-B", "args": (y, yprev, yprev2, fit_y), "bounds": bounds}
    k_opt = sp.optimize.basinhopping(findBestAR, x0 = x0_vals, minimizer_kwargs = k_kwargs,
                    niter = 10)
    
    if do_print:
        print(k_opt.x)
        
    k1 = k_opt.x[0]                  
    res_post = y[2:] - (1 - k1) * yprev[1:] - (1 - k2) * yprev2 - fit_y[2:] * (k1 + k2 - 1)
    calc_rat_val_AR = -np.sum(np.log(ratioProb.pdf(res_post, k_opt.x[3], k_opt.x[1], k_opt.x[2])))
    rep_rat_val_AR = k_opt.fun * (len(y) - 2)
    fun_prop_diff = (calc_rat_val_AR - rep_rat_val_AR) * 2 / (calc_rat_val_AR + rep_rat_val_AR)
    print('final reported fun value: ', rep_rat_val_AR)
    print('final calculated fun value: ', calc_rat_val_AR)
    if fun_prop_diff > 0.02:
        print('error: calculated and reported function values different')
        wait = input("PRESS ENTER TO CONTINUE.")
    return k_opt.x

def regress_numeric_line_opt(line_pars, x, fit_y, pars, k1, k2 = 1):
    fit_y = np.array(fit_y)
    y = line_pars[1] + line_pars[0] * x
    yprev = y[:-1]
    yprev2 = y[:-2]
    res = y[2:] - (1 - k1) * yprev[1:] - (1 - k2) * yprev2 - fit_y[2:] * (k1 + k2 - 1)
    ent = -np.sum(np.log(ratioProb.pdf(res, pars[2], pars[0], pars[1])))
    return ent

def max_likelihood_line(xs, ys, line_pars):
    # maximum likelihood of y, given fitted line, rather than add adj                       
    mytakestep = MyTakeStep()
    
    (a, b, r, tt, stderr) = sp.stats.linregress(xs, ys)
    if a > 0:
        xmin = [a * 0.5, b * 0.9]
        xmax = [a * 2, b * 1.1]
    else:
        xmin = [a * 2, b * 0.9]
        xmax = [a * 0.5, b * 1.1]
    bounds = [(low, high) for low, high in zip(xmin, xmax)]
    
    y_pars = line_pars[1].x
    y_fit = y_pars[0] * xs + y_pars[1]               
    kwargs = {"method": "L-BFGS-B", "args": (xs, y_fit, line_pars[5], line_pars[4]), "bounds": bounds}
    par_opt = sp.optimize.basinhopping(regress_numeric_line_opt, x0 = [a, b], minimizer_kwargs = kwargs,
                niter = 20, take_step = mytakestep)
    return par_opt.x

def predict_true_analy(ce, disc, line_grad, post_chg_arr4):
    true_predict_corr = []; sign_corr = []; med_true_predict_abs = []; med_true_predict_rel = [];
    for l in range(1, 25, 1):
        predict = []; true = []; abs_diff = []; rel_diff = []; 
        for m in range(0, int(len(disc) - l - ce), 2):
            start = m
            end = int(m + l)
            cent = int(m + l + ce)
            cor_res = robust_cor([list(disc[start : end]), line_grad[start : end]], post_chg_arr4[start : end])
            coeffs = [*cor_res[1], cor_res[2]]
            cur_predict = coeffs[0] * disc[cent] + coeffs[1] * line_grad[cent] + coeffs[2]
            predict.append(cur_predict)
            cur_true = post_chg_arr4[cent]
            true.append(cur_true)
            abs_diff.append(cur_true - cur_predict)
            rel_diff.append((cur_true - cur_predict) / cur_true)
        true_predict_corr.append(robust_cor(predict, true)[0])
        sign_corr.append(robust_cor(np.sign(predict), np.sign(true))[0])
        med_true_predict_abs.append(np.median(abs_diff))
        med_true_predict_rel.append(np.median(rel_diff))
    return (true_predict_corr, sign_corr, med_true_predict_abs, med_true_predict_rel)

def robust_cor(x, y):
    if isinstance(x[0], list):
        x = list(map(list, zip(*x)))
    else:
        x = np.array(x).reshape(-1, 1)
    X = np.array(x)
    Y = np.array(y)
    theil_regr = TheilSenRegressor(random_state = 42)
    theil_regr.fit(X, Y)
    y_pred = theil_regr.predict(X)
    res = y_pred - y
    tot_dev = y - np.mean(y)
    SSres = np.dot(res, res)
    SStot = np.dot(tot_dev, tot_dev)
    adjR2 = 1 - (SSres/SStot) * (X.shape[0] - 1) / (X.shape[0] - X.shape[1] - 1)
    sgn = np.sign(theil_regr.coef_)[0]
    if adjR2 > 0:
        corr_val = sgn * np.sqrt(adjR2)
    else:
        corr_val = 0
    return [corr_val, theil_regr.coef_, theil_regr.intercept_, theil_regr.breakdown_]

def stck_pricing(latest_y, pre_len, pricing, ahead4_val, disc, line_grad, post_chg_arr4):
    lst_pre_ind = np.array(*np.where(ahead4_val == True))[-1]
    fst_pre_ind = lst_pre_ind - pre_len
    new_val = lst_pre_ind + 5
    print(disc[-1], disc[new_val])
    cor_res = robust_cor([list(disc[fst_pre_ind : lst_pre_ind + 1]), line_grad[fst_pre_ind : lst_pre_ind + 1]], post_chg_arr4[fst_pre_ind : lst_pre_ind + 1])
    coeffs = [*cor_res[1], cor_res[2]]
    pred_chg4 = coeffs[0] * disc[new_val] + coeffs[1] * line_grad[new_val] + coeffs[2]
    # adjust for latest price
    # get predicted price, not just change
    pricing.append(pred_chg4)
    return pricing

def prepare_pricing(latest_y, disc, line_grad, post_chg_arr4, lst_disc, lst_line_grad, pricing):
    lst_disc.append(disc[-1])
    lst_line_grad.append(line_grad[-1])
    
    np_disc = np.array(disc); np_line_grad = np.array(line_grad); np_post_chg_arr4 = np.array(post_chg_arr4)
    ahead4_val = np.isfinite(np_post_chg_arr4)
    do_prc_analysis = False
    
#    if do_prc_analysis and np.sum(ahead4_val) > 6:
#        [true_predict_corr, sign_corr, med_true_predict_abs, med_true_predict_rel] \
#            = predict_true_analy(4, np_disc[ahead4_val], np_line_grad[ahead4_val], np_post_chg_arr4[ahead4_val])
#        if (np.max(true_predict_corr) > 0.25):
#            pre_len = np.argmax(true_predict_corr)
#            pricing = stck_pricing(latest_y, pre_len, pricing, ahead4_val, disc, line_grad, post_chg_arr4)
#        else:
#            pricing.append(np.nan)
#            
#    else:
    pricing = stck_pricing(latest_y, 5, pricing, ahead4_val, disc, line_grad, post_chg_arr4)
        
    return (lst_disc, lst_line_grad, pricing)

def best_price(pricing, stock, lst_disc, lst_line_grad):
    max_ind = np.argmax(np.abs(pricing))
    arb = pricing[max_ind]
    arb_stk = stock[max_ind]
    if arb > 0:
        print('Buy')
        direc = 'underpriced'
    elif arb < 0:
        print('Sell')
        direc = 'overpriced'
    print(arb_stk, ' is maximum ', direc, ' by ', arb)
    print(lst_disc[max_ind], lst_line_grad[max_ind])
    # plt.plot to show more?
    print()
    return (arb_stk, arb)

def main():
    
    initialise = False
    if initialise:
        init()
    
    size = 200
    x = np.linspace(0, 2, size)
    datafiles = ['NVDA_2Daily_150']  # 'FB_2Daily_300'  'NVDA_2Daily_150'  'AAPL_2017_1_15' 'FB_2017_1_15', 
    pricing = []
    stock = []
    lst_disc = []; lst_line_grad = []
                
    for data_file in datafiles:
        dfAAPL = pd.read_pickle(data_file)
        stock.append(data_file.split('_')[0])
        #dfAAPL_res3D = dfAAPL.resample('3D').mean()
        #x = dfAAPL.index.astype(np.int64) / 1e10
        y = 50 * dfAAPL['logMedian']  # 50 * improves numerical solver
        
        disc = []; line_grad = []; post_chg_arr4 = []
        line_pars = []; line_pars0 = []
        for mu_l in [1, 2]:  #   * std
            for sampleSz in [20]:
                xs = x[0:sampleSz]
                g = 1
                start = len(dfAAPL) - g - sampleSz
                finish = start + sampleSz           
                save_file = 'line_pars_' + stock[-1] + '_2017_1_15_inc1_std' + str(mu_l) \
                            + '_samp' + str(sampleSz) + '_2daily'                            
                if os.path.isfile(save_file):
                    exist_pars = pd.read_pickle(save_file).values.tolist()
                    cur_lst = exist_pars[0][6][1]
                    upd_exist = True
                    print('Updating existing line data file')
                else:
                    max_pers = 2 #150
                    upd_exist = False
                    print('Creating new line data file')
                
                if upd_exist:
                    do = (start > 0) and (dfAAPL.index[finish - 1] > cur_lst)
                else:
                    do = (start > 0) and (g < max_pers + 1)
                
                while do:
                    ys = list(y[start : finish])
                    print('sample ', str(g - 1), ': processing ', finish - start, ' data pts')
                    line_fit = AR_line_calc(xs, ys, mu_l)
                    par_vals = regress_numeric_line_AR_ent_final(line_fit[1].x, xs, ys, False, mu_l)
                    line_fit[4] = par_vals[0]
                    line_fit[5] = par_vals[1:]
                    startdt = dfAAPL.index[start] # .to_pydatetime()
                    finishdt = dfAAPL.index[finish - 1]
                    line_fit.append([startdt, finishdt])
                    if (mu_l == 1):
                        line_pars0.append(line_fit)
                    elif (mu_l == 2):
                        maxl_y_pars = max_likelihood_line(xs, ys, line_fit)
                        y_max_line = maxl_y_pars[0] * xs + maxl_y_pars[1]
                        line_fit.append(y_max_line)
                        if not(abs(line_fit[1].x[0] - maxl_y_pars[0]) < 0.01):
                            print('error: line grads differ') # then may need to store maxl_y_pars
                        # forward vals
                        next_10_end = min(finish + 10, len(dfAAPL))
                        next_10_start = min(finish, len(dfAAPL))
                        next_10 = np.array(y[next_10_start : next_10_end])
                        line_fit.append(next_10)
                        # OLS fit for outlier removal
                        (a, b, r, tt, stderr) = sp.stats.linregress(xs, ys)
                        ols_fit = b + a * xs
                        res_ols = ys - ols_fit
                        std_res_ols = np.std(res_ols)
                        ols_val = -np.sum(np.log(sp.stats.norm.pdf(res_ols, np.mean(res_ols), std_res_ols))) \
                                * (len(ys) - 2) / len(ys)
                        line_fit.append(ols_val)
                        line_pars.append(line_fit)
                    print('finished')
                    print()
                    g = g + 1
                    start = start - 1
                    finish = start + sampleSz
                    if upd_exist:
                        do = (start > 0) and (dfAAPL.index[finish - 1] > cur_lst)
                    else:
                        do = (start > 0) and (g < max_pers + 1)
                    
                try:
                    if upd_exist:
                        if (mu_l == 1):
                            line_pars0.extend(exist_pars)
                            pd.DataFrame(line_pars0).to_pickle(save_file)
                        elif (mu_l == 2):
                            for line in exist_pars:
                                finish_ind = np.where(dfAAPL.index == line[6][1])[0]
                                if finish_ind.size > 0:
                                    next_10_end = min(finish_ind[0] + 11, len(dfAAPL))
                                    next_10_start = min(finish_ind[0] + 1, len(dfAAPL))
                                    next_10 = np.array(y[next_10_start : next_10_end])
                                    line[8] = next_10
                            line_pars.extend(exist_pars)
                            pd.DataFrame(line_pars).to_pickle(save_file)
                except:
                    print('error: save error')
                
                if (mu_l == 2):
                    # remove outliers
                    std_fun_diff = [x[1].fun - y[1].fun for x, y in zip(line_pars, line_pars0)]
                    #ols_fun_diff = [x[1].fun * (len(x[0]) - 2) - x[9] for x in line_pars]
                    
                    outliers = np.where(np.array(std_fun_diff) > 0.01)[0]
                    #outliers = np.append(outliers, np.where(np.array(ols_fun_diff) > 0.01)[0])
                    outliers = list(set(outliers))
                    print('Removing', "{:.0f}".format(100 * len(outliers) / len(line_pars)), '% as outliers')
                    if (len(line_pars) - 1) in outliers:
                        print('Current line is outlier. Quitting')
                        return
                    #line_pars = np.delete(line_pars, outliers)
                    
                    disc = [(x[0][-1] - x[7][-1]) for x in line_pars]
                    line_grad = [x[1].x[0] for x in line_pars]
                    post_chg_arr4 = [(x[8][4] - x[0][-1]) if len(x[8]) > 4 else np.nan for x in line_pars]
                    #get ys[-1] and latest_y
                    [lst_disc, lst_line_grad, pricing] = prepare_pricing(5.345, disc, line_grad, post_chg_arr4, lst_disc, lst_line_grad, pricing)
    
    [arb_stk, arb] = best_price(pricing, stock, lst_disc, lst_line_grad)
    # then manually act in TWS?
    

if __name__ == "__main__":
    main()
    
    