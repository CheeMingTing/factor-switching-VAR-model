function [Q, cov_f, f, cov_e, e] = fmest(y,r)
%==========================================================================
%  Estimation of factor model using basic principal component analysis (PCA)                                                
%    
%    Model: y(t) = Q*f(t)   + e(t)  e(t)~N(0,cov_e) t=1...T
%              
%              N  - number of time series
%              T  - length of time series
%              r  - number of factors
%            y(t) - N*1 observed vector of time series
%            f(t)- r*1 vector of latent factor time series  with (unknown) r<N
%             Q   - N*r factor loading matrix, Q'Q = I
%
%   Input:    y  - N*T time series data
%             r  - known number of factors
%   Output:   Estimators of Q, f, cov_e, e (estimated residuals)
%
%   Author: Chee-Ming Ting, Universiti Teknologi Malaysia & KAUST
%
%   Reference: 
%   C. A. Favero, "Principal components at work: the empirical analysis of
%   monetary policy with large data sets," Journal of Applied Econometrics,
%   vol. 20, pp. 603-620, 2005.
%==========================================================================
[N,T] = size(y);
Y = y';
LQ = Y'*Y / (T*N);
[UQ,DQ] = eig(LQ);
UQ = UQ(:, end:-1:1);
DQ = DQ(end:-1:1,end:-1:1);
%     Q  = UQ(:,1:r) * sqrt(N);
%     F  = Y*Q / N;  f = F';
Q  = UQ(:,1:r);
F  = Y*Q;  f = F'; % Use normalization Q'Q = I
e = y - Q*f;  
cov_f = F'*F / T;
cov_e = cov(e'); % Estimated sample noise covariance from residuals
