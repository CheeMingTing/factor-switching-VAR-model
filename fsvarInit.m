function [fsvar0,fm] = fsvarInit(Y,T,opts)
%--------------------------------------------------------------------------
% Title    : Parameter Initialization
% Function : Initialize parameters of f-SVAR model for EM estimation
%            based on estimates of common factor model on concatenated data
%               (1) Estimate a factor model by PCA
%               (2) Initialize VAR coeff. matrix at each state based on 
%                   K-means clustering of sliding-windowed VAR coefficients 
%                   estimated from factors in step (1)
% Input:
%       Y - N X T time series data (concatenated across subjects)
%       T - Total number of time points
%       opts: structure with model specification
%               p - VAR model order
%               K - Number of states
%               r - Number of factors
%               wlen - window size
%               shift - window shift
% Output:
%       fsvar0: Struct containing initial estimates of f-SVAR model parameters
%               A - VAR coeff. mat
%               V - State cov. mat
%               H - Obs mapping
%               R - Obs cov. mat
%               x0 - Initial state
%               Z,pi - State transition matrix
%       fm: Struct containing estimates of factor model
%               Q - factor loading matrix
%               f - factor time series
%               e - residuals
%               cov_f - covariance matrix of factors
%               cov_e - residual covariance matrix
% Author: Chee-Ming Ting, UTM, KAUST Nov 2017
%--------------------------------------------------------------------------
p = opts.p;
K = opts.K;
r = opts.r;
[N,~] = size(Y);
A0 = zeros(r*p,r*p,K);
V0 = zeros(r*p,r*p,K);
H0 = zeros(N,r*p,K);
R0 = zeros(N,N,K);
x0 = zeros(r*p,1);

%---------------------------------------------------------------------%
%               Estimate common factor model                          %
%---------------------------------------------------------------------%
[Q, cov_f, f, cov_e, e] = fmest(Y,r); % PCA estimation of fm

%---------------------------------------------------------------------%
%               Initialize f-SVAR parameters                          %
%---------------------------------------------------------------------%
% Initialize A by K-means clustering of factor var coeffs.
wlen = opts.wlen;
shift = opts.shift;
tvvar_vec = zeros(p*r^2,T);
fw   = zeros(r,wlen);
win = rectwin(wlen);

% Sliding-window analysis
indx = 0; t = 1;
while indx + wlen <= T
      for i=1:r
          fw(i,:) = f(i,indx+1:indx+wlen).*win'; end 
      [Aft,Vt] = varfit(p,fw);
      tvvar_vec(:,t) = Aft(:); % Time-varying var
      indx = indx + shift; t = t + 1;
end
fprintf('Initialize by K-means clustering... \n\n');
optkm = statset('Display','final');
[fSt_km,~,~,~] = kmeans(tvvar_vec',K,'Distance','cityblock','Replicates',100,'start','cluster','Options',optkm);

% Fit state-specific factor var
Ft = zeros(r,T,K);
A_km_f = zeros(r,r*p,K);
tj = zeros(K,1);
for j=1:K
    t=1;
    for i=1:T
        if fSt_km(i) == j
           Ft(:,t,j) = f(:,i);  t=t+1; end
    end
    tj(j) = t-1;
    [A_km_f(:,:,j),~] = varfit(p,Ft(:,1:tj(j),j));
    A0(1:r,1:r*p,j) = A_km_f(:,:,j);
    for k=1:p
    if k<p
       A0(k*r+1:k*r+r,(k-1)*r+1:(k-1)*r+r,j) = eye(r); end
    end
end

% Other parameters
for j=1:K
    V0(:,:,j) = eye(r*p);
    H0(1:N,1:r,j) = Q; % From PCA-estimated factor loadings
    R0(:,:,j) = eye(N);
end

% Markov transition matrix
pi   = ones(1,K)/K;
Z0   = 0.05/(K-1)*ones(K,K); 
Z0(1:K+1:end) = 0.95;

% Output
fsvar0.A = A0; fsvar0.V = V0; fsvar0.H = H0; fsvar0.R = R0;
fsvar0.x0 = x0; fsvar0.Z = Z0; fsvar0.pi = pi; 
% fsvar0.K = K; fsvar0.p = p;
fm.Q = Q; fm.cov_f = cov_f; fm.f = f; fm.cov_e = cov_e; fm.e = e;
end

