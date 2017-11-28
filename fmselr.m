function [r, b, bic] = fmselr(y,kmax)
%==========================================================================
%  Determining number of factors r in factor model from canditates [1 kmax]
%
%    Model:
%            y(t) = Q*f(t) + e(t)  e(t)~N(0,cov_e) t=1...nT
%              
%              N  - number of time series
%              T  - number of time points
%              r  - number of factors
%            y(t) - N*1 observed vector of time series
%            f(t)- r*1 vector of latent factor time series with(unknown)r<N
%             Q   - N*r factor loading matrix, Q'Q = I
%
%   Input:    y   - N X T observations, kmax - maximum range of candidate r
%   Output:   r   - Optimal number of factors based on BIC
%
%   Author: Chee-Ming Ting, , Universiti Teknologi Malaysia & KAUST (2017)
%
%   Reference: 
%   J. Bai & S. Ng, "Determining the number of factors in approximate
%   factor models," Econometrica, vol. 70, no. 1, pp. 191-221, 2002.
%
%==========================================================================
[N,T] = size(y);

for k=1:kmax
    [Q, cov_f, f, cov_e, e] = fmest(y,k);
    bic(k,1) = log(sum(sum(e.^2))/(N*T)) + k * ((N+T)/(N*T)) * log((N*T)/(N+T));   
end
[b,r] = min(bic);