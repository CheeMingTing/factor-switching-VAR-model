function [fsvar,Path,L] = fsvarest(Yr,fsvar0,opts)
%--------------------------------------------------------------------------
% Title    : Estimation of f-SVAR model by Expectation-maximization (EM)
% Function : Estimate hidden state sequence by Kalman filtering and
%            smoothing, and update model parameters with EM algorithm
% 
% Inputs:    Yr - N x T x Rs time series data, N = d_obs
%            fsvar0: struct with initialized parameters
%                   A - VAR coeff matrix (state transition x(t-1)->x(t))
%                   V - State noise covariance matrix
%                   H - Observation mapping matrix of x(t)->y(t)
%                   R - Observation noise covariance matrix                   
%                   x0 - Initial state
%                   Z - Markov transition matrix 
%                   pi - Initial state probability
%                   K  - Number of regimes/states
%            opts: struct of EM algorithm settings
%                   ItrNo - maximum interation of EM algorithm
%                   eps   - Stop EM when eps< likelihood improvement
%                   
% Outputs:
%   Path: Structure with sequence of estimated latent regimes St and factor
%         dynamics, xt for each replicate s
%         Rs  - number of replicates
%         fSt - State/Regime probability estimated by switching Kalman filter, P(St=j|y(1:t)) 
%         sSt - State/Regime probability estimated by switching Kalman smoother, P(St=j|y(1:T))
%         St_skf - Most likely state sequence by hard assignment, argmax_j P(St=j|y(1:t))
%         St_sks - argmax_j P(St=j|y(1:T))
%         fxt - Filtered state vector, P(xt|y(1:t)) 
%         sxt - Smoothed state vector, P(xt|y(1:t))
%         fft - Filtered factors (fxt(1:r,:,:) : r X 1 vectors of factor components)
%         sft - smoothed factors (sxt(1:r,:,:)
%   fsvar: Struct with EM estimated model parameters
%         A[:,:,j]  - AR coefficient matrix at regime j
%         V[:,:,j]  - State-specific noise cov at regime j
%   L  - Log-likelihood each iterations
%                    
%  Variables:
%         Rs = number of signal replicates (or subjects)
%         T  = length of each replicate
%         d_obs = dimension of observation vector
%         d_state = dimension of state vector
%
% Authors: Chee-Ming Ting, Universiti Teknologi Malaysia & KAUST (2017)
%          Siti Balqis Samdin, Universiti Teknologi Malaysia
%
% Reference: [1] S. B. Samdin, C.-M. Ting, H. Ombao, and S.-H Salleh,
%           “A Unified estimation framework for state-related changes
%           in effective brain connectivity,” IEEE Trans. Biomed. Eng. 2017
%            [2] K. Murphy, “Switching kalman filters,” 1998.
%--------------------------------------------------------------------------

% Initialize parameters
A = fsvar0.A; Q = fsvar0.V; H = fsvar0.H; R = fsvar0.R;
x0 = fsvar0.x0; Z = fsvar0.Z; pi = fsvar0.pi;

[d_obs,T,Rs] = size(Yr);
y = zeros(T, d_obs);
d_state = size(A,1);
M = size(Z,1);
I = eye(d_state);

% Declaring Kalman filter variables
x_ij = zeros(d_state,M,M);
P_ij = zeros(d_state,d_state,M,M);
xhat = zeros(d_state,M,T);
Phat = zeros(d_state,d_state,M,T);
xx_minus = zeros(d_state,M,M,T);
PP_minus = zeros(d_state,d_state,M,M,T);
Pe = zeros(d_obs,d_obs,M,M,T);
e  = zeros(d_obs,M,M,T);
L = zeros(M,M,T);
S = zeros(T,M);     
S_MtT = zeros(T,M);
fSt = zeros(T,M,Rs);
fxt = zeros(d_state,T,Rs);

% Declaring Kalman smoothing variables
xshat = zeros(d_state,M,T);
Pshat = zeros(d_state,d_state,M,T);
Jt = zeros(d_state,d_state,M,M,T);
xs = zeros(d_state,M,M);
Ps = zeros(d_state,d_state,M,M);
xs_t = zeros(d_state,M,M,T);
Ps_t = zeros(d_state,d_state,M,M,T);
P_ttm1T = zeros(d_state,d_state,M,M,T);
sSt = zeros(T,M,Rs);
sxt = zeros(d_state,T,Rs);

% Declaring E-step variables
S_t    = zeros(d_state,d_state,T);
S_ttm1 = zeros(d_state,d_state,T);

% EM iteration
fprintf('\nEM estimation... \n');
OldL = 0;
for it=1:1:opts.ItrNo

x = zeros(d_state,M,T);
P = zeros(d_state,d_state,M);
x(:,:,1) = repmat(x0,1,M); % Initialize state parameter
P0 = eye(d_state);
for i=1:M
  P(:,:,i) = P0; end
S(1,:) = pi;

% Accumulated statistics over all replicates
SumA1R = zeros(d_state,d_state,M);
SumA2R = zeros(d_state,d_state,M);
SumQ1R = zeros(d_state,d_state,M);
SumQ2R = zeros(d_state,d_state,M);
SumWR = zeros(1,M);
Lt = 0;

for s = 1:Rs

y(:,:) = squeeze(Yr(:,:,s))';
    
P_ttm1T_full = zeros(d_state,d_state,T);
Pshat_full = zeros(d_state,d_state,T);
xshat_full = zeros(d_state,T);
Phat_full  = zeros(d_state,d_state,T);
xhat_full  = zeros(d_state,T);

try
%-------------------------------------------------------------------------%
%                        Switching Kalman Filter                          %
%-------------------------------------------------------------------------%   
for t=2:T
    S_norm = 0;
    S_marginal = zeros(M,M);

  for j=1:M
      A_j = A(:,:,j);
      H_j = H(:,:,j);
      Q_j = Q(:,:,j);
      R_j = R(:,:,j);

    for i=1:M
      % One-step ahead Prediction
      x_minus = A_j * x(:,i,t-1);
      P_minus = A_j * P(:,:,i) * A_j' + Q_j;      
      xx_minus(:,i,j,t)   = x_minus;
      PP_minus(:,:,i,j,t) = P_minus;
      
      % Prediction error
      Pe(:,:,i,j,t) = H_j * P_minus * H_j' + R_j;
      e(:,i,j,t)    = y(t,:)' - H_j * x_minus;  
      
      % Kalman Gain
      K = P_minus * H_j' * pinv(Pe(:,:,i,j,t));
      
      % Filtering update
      x_ij(:,i,j)   = x_minus + K * e(:,i,j,t);
      P_ij(:,:,i,j) = (I - K*H_j)*P_minus;
      
      if t==T
         P_ttm1T(:,:,i,j,t) = (I-K*H_j)*A_j*Phat(:,:,j,t-1);
      end

      % Predictive Likelihood L(i,j,t) = P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
       a = squeeze(e(:,i,j,t)'); covar = squeeze(Pe(:,:,i,j,t));
       L(i,j,t) = (det(covar))^-.5 * exp(-.5 * sum(((a*pinv(covar)).*a), 2));

      S_marginal(i,j) = L(i,j,t) * Z(i,j) * S(t-1,i);
      S_norm = S_norm + S_marginal(i,j);
      
      clear x_minus P_minus a covar K;

    end   
      clear A_j H_j Q_j R_j
   end

  % Filtered occupancy probability of state j at time t
  S_marginal = S_marginal/S_norm; % P(S(t)=j,S(t-1)=i|y(1:t))
  for j=1:M
      S(t,j) = sum(S_marginal(:,j)); % P(S(t)=j|y(1:t))
  end
       
  % Weights of state components
  W = zeros(M,M);
  for j=1:M
      for i=1:M
          W(i,j) = S_marginal(i,j)/S(t,j); % P(S(t-1)=i|S(t)=j,y(1:t))
      end
  end

  % Collapsing: Gaussian approximation
  for j=1:M
      x(:,j,t) = x_ij(:,:,j) * W(:,j);
      P(:,:,j) = zeros(d_state,d_state);
      for i=1:M
          m = x_ij(:,i,j) - x(:,j,t);
          P(:,:,j) = P(:,:,j) + W(i,j)*(P_ij(:,:,i,j) + m*m');
          clear m;
      end 
      % Filtered density of x(t) given state j
      xhat(:,j,t) = x(:,j,t);   % E(x(t)|S(t)=j,y(1:t))
      Phat(:,:,j,t) = P(:,:,j); % Cov(x(t)|S(t)=j,y(1:t))
  end
  
  % Filtered density of x(t)
    for j=1:M
        xhat_full(:,t) = xhat_full(:,t) + xhat(:,j,t) * S(t,j); % E(x(t)|y(1:t))
    end
    for j=1:M
        mu = xhat(:,j,t) - xhat_full(:,t);
        Phat_full(:,:,t) = Phat_full(:,:,t) + S(t,j)*(Phat(:,:,j,t) + mu*mu');  % Cov(x(t)|y(1:t))
    end
    clear S_marginal W;
end % End for t=2:T

% Filtered state sequence
fSt(2:T,:,s) = S(2:T,:);
fxt(:,2:T,s) = xhat_full(:,2:T);

catch
    fprintf('sub-%d\n',s);
end

%-------------------------------------------------------------------------%
%                        Switching Kalman Smoother                        %
%-------------------------------------------------------------------------%
    S_MtT(T,:) = S(T,:);
    xshat(:,:,T)   = xhat(:,:,T);
    Pshat(:,:,:,T) = Phat(:,:,:,T);  
    xshat_full(:,T)   = xhat_full(:,T);
    Pshat_full(:,:,T) = Phat_full(:,:,T);
    S_Mttp1T = zeros(M,M,T);
            
for t=T-1:-1:1
    S_n = zeros(M,1);
    S_m = zeros(M,M);
   
    for k=1:M
        A_k = A(:,:,k);                   
        for j=1:M
           Jt(:,:,j,k,t) = Phat(:,:,j,t) * A_k' * pinv(PP_minus(:,:,j,k,t+1)); %J(t)
           xs(:,j,k) = xhat(:,j,t) + Jt(:,:,j,k,t)*(xshat(:,k,t+1) - A_k*xx_minus(:,j,k,t+1)); %X(t|T)
           Ps(:,:,j,k) = Phat(:,:,j,t) + Jt(:,:,j,k,t)*(Pshat(:,:,k,t+1) - PP_minus(:,:,j,k,t+1)) * Jt(:,:,j,k,t)';    %V(t|T)
           xs_t(:,j,k,t) = xs(:,j,k);  Ps_t(:,:,j,k,t) = Ps(:,:,j,k);    
           S_m(j,k) = S(t,j) * Z(j,k);
        end
    end

    for k=1:M
        for j=1:M
            S_n(k,1) = S_n(k,1) + S_m(j,k); end
    end
    
    for k=1:M
        for j=1:M
            U(j,k) = S_m(j,k)/S_n(k,1);
            U_t(j,k,t) = U(j,k);
        end
    end

    for k=1:M
        for j=1:M
            S_Mttp1T(j,k,t+1) = U(j,k)*S_MtT(t+1,k); end
    end
    
    % Smoothed occupancy probability of state j at time t
    for j=1:M
        S_MtT(t,j) = sum(S_Mttp1T(j,:,t+1));
    end
    for j=1:M
        for k=1:M
            W_2(k,j)= S_Mttp1T(j,k,t+1)/S_MtT(t,j); % P(S(t+1)=k|S(t)=j,y(1:T))
        end
    end
    
    % Collapsing
    xshat_j = zeros(d_state,M);
    Pshat_j = zeros(d_state,d_state,M);
    for j=1:M
        for k=1:M
            xshat_j(:,j) = xshat_j(:,j) + xs(:,j,k) * W_2(k,j);
        end
        for k=1:M
            m2 = xs(:,j,k) - xshat_j(:,j);
            Pshat_j(:,:,j) = Pshat_j(:,:,j) + W_2(k,j)*(Ps(:,:,j,k) + m2*m2');
            clear m2;
        end 
        % Smoothed density of x(t) given state j
        xshat(:,j,t)   = xshat_j(:,j);     % E(x(t)|S(t)=j,y(1:T))    (Eq. 13)
        Pshat(:,:,j,t) = Pshat_j(:,:,j);   % Cov(x(t)|S(t)=j,y(1:T))  (Eq. 14)
    end
    
    % Smoothed density of x(t)
    for j=1:M
        xshat_full(:,t) = xshat_full(:,t) + xshat_j(:,j) * S_MtT(t,j); % E(x(t)|y(1:T))
    end
    for j=1:M
        m3 = xshat_j(:,j) - xshat_full(:,t);
        Pshat_full(:,:,t) = Pshat_full(:,:,t) + S_MtT(t,j)*(Pshat_j(:,:,j) + m3*m3'); % Cov(x(t)|y(1:T))
        clear m3;
    end
end

% Smoothed state sequence
sSt(1:T,:,s) = S_MtT(1:T,:);
sxt(:,2:T,s) = xshat_full(:,2:T);

% Cross-variance terms
for t=(T-1):-1:2
    for k=1:M
        A_k = A(:,:,k);
        for j=1:M
           P_ttm1T(:,:,j,k,t) = Phat(:,:,j,t)*Jt(:,:,j,k,t-1)'+Jt(:,:,j,k,t)...
               *(P_ttm1T(:,:,j,k,t+1)-A_k*Phat(:,:,j,t))*Jt(:,:,j,k,t-1)';  %V(t,t-1|T)_jk
        end
    end
end
% Cross-collapsing cross-variance
for t=T:-1:2
    for k=1:M
        mu_x = 0; mu_y = 0; 
        P_ttm1T_k(:,:,k,t) = zeros(d_state,d_state);
        for j=1:M
             mu_x = mu_x + xshat(:,k,t)*U_t(j,k,t-1);
             mu_y = mu_y + xs_t(:,j,k,t-1)*U_t(j,k,t-1);
        end
        mu_y_k(:,k,t) = mu_y;
        for j=1:M
            P_ttm1T_k(:,:,k,t) = P_ttm1T_k(:,:,k,t) + U_t(j,k,t-1)*(P_ttm1T(:,:,j,k,t) + (xshat(:,k,t)-mu_x)*(xs_t(:,j,k,t-1)-mu_y)'); %(Eq. 15)
        end
        clear mu_x mu_y;
    end
    
    mu_x = 0; mu_y = 0; 
    for k=1:M
        mu_x = mu_x + xshat(:,k,t) * S_MtT(t,k);
        mu_y = mu_y + mu_y_k(:,k,t-1)  * S_MtT(t,k);
    end
    for k=1:M
        P_ttm1T_full(:,:,t) = P_ttm1T_full(:,:,t) + S_MtT(t,k)*(P_ttm1T_k(:,:,k,t) + (xshat(:,k,t)-mu_x)*(mu_y_k(:,k,t-1)-mu_y)');
    end
    clear mu_x mu_y;
end

%-------------------------------------------------------------------------%
%                        Log-likelihood computation                       %
%-------------------------------------------------------------------------%
% P(y(t)|y(1:t-1) = sum_i sum_j [P(y(t),S(t)=j,S(t-1)=i|y(1:t-1))]
% where P(y(t),S(t)=j,S(t-1)=i|y(1:t-1) =
% P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)*P(S(t)=j|S(t-1)=i,y(1:t-1))*P(S(t-1)=i|y(1:t-1))
% = L(i,j,t)*Z(i,j)*S(t-1,j) = S_marginal(i,j)
% Lt = 0;
for t=2:T
    Acc = 0;
    for j=1:M
        for i=1:M
            log_S_marg_ij = - 0.5*(log(det(Pe(:,:,i,j,t))) - 0.5*e(:,i,j,t)'*pinv(Pe(:,:,i,j,t))*e(:,i,j,t)) + log(Z(i,j)) + log(S(t-1,i));
            Acc = Acc + exp(log_S_marg_ij);
        end
    end
    Lt = Lt + log(Acc);
end

%-------------------------------------------------------------------------%
%                               E-step                                    %
%-------------------------------------------------------------------------%
for t=2:T
    S_t(:,:,t)    = Pshat_full(:,:,t)   + xshat_full(:,t)*xshat_full(:,t)';       % (Eq. 18)
    S_ttm1(:,:,t) = P_ttm1T_full(:,:,t) + xshat_full(:,t)*xshat_full(:,t-1)';     % (Eq. 19)
end

%-------------------------------------------------------------------------%
%                               M-step                                    %
%-------------------------------------------------------------------------%
SumA1 = zeros(d_state,d_state,M);
SumA2 = zeros(d_state,d_state,M);
SumQ1 = zeros(d_state,d_state,M);
SumQ2 = zeros(d_state,d_state,M);
SumW = zeros(1,M);

Wj_t = S_MtT(1:T,:);

for i=1:M
    SumW(1,i) = sum(Wj_t(2:T,i));
    for t=2:T
       SumA1(:,:,i) = SumA1(:,:,i) + Wj_t(t,i)*S_ttm1(:,:,t);
       SumA2(:,:,i) = SumA2(:,:,i) + Wj_t(t,i)*S_t(:,:,t-1);
       SumQ1(:,:,i) = SumQ1(:,:,i) + Wj_t(t,i)*S_t(:,:,t);
       SumQ2(:,:,i) = SumQ2(:,:,i) + Wj_t(t,i)*S_ttm1(:,:,t)';
    end   
end

for i=1:M
    SumWR(1,i) = SumWR(1,i) + SumW(1,i);
    SumA1R(:,:,i) = SumA1R(:,:,i) + SumA1(:,:,i);
    SumA2R(:,:,i) = SumA2R(:,:,i) + SumA2(:,:,i);
    SumQ1R(:,:,i) = SumQ1R(:,:,i) + SumQ1(:,:,i);
    SumQ2R(:,:,i) = SumQ2R(:,:,i) + SumQ2(:,:,i);
end

clear SumA1 SumA2 SumQ1 SumQ2;
clear P_ttm1T_full Pshat_full xshat_full Phat_full xhat_full

end % over replicates

Qa = zeros(d_state,d_state);
for i=1:M
    A(:,:,i)= SumA1R(:,:,i)*pinv(SumA2R(:,:,i));
    Qa(:,:) = (1/SumWR(1,i))*(SumQ1R(:,:,i) - A(:,:,i)*SumQ2R(:,:,i));
    % Constrait A for VAR(p) factors
    r = opts.r;
    A(r+1:d_state,1:d_state,i) = 0;
    for k=1:opts.p
        if k<opts.p
            A(k*r+1:k*r+r,(k-1)*r+1:(k-1)*r+r,i) = eye(r); end
    end
%     Q(:,:,i) = Qa(:,:); % Unconstrained - Full matrix
    Q(:,:,i) = diag(diag(Qa(:,:))); % Contrained - diagonal matrix
end
clear SumA1R SumA2R SumQ1R SumQ2R Qa;

L(it) = Lt;
if (it>1) 
    fprintf('Iteration-%d   Log-like = %g\n',it,L(it)); end
DeltaL = (L(it)-OldL)/L(it);DeltaL=abs(DeltaL); % Stoping Criterion (Relative Improvement)
if(DeltaL < opts.eps)
%     fprintf('\nEM reestimation complete\n\n');
    ConvergeL = L(it);
    fprintf('Converge log-like = %g\n\n',ConvergeL);
    break;
end
OldL = L(it);
end % over iterations

% Compute most likely state sequence
St_skf = zeros(T,Rs);
St_sks = zeros(T,Rs);
for s =1:Rs
    [~, St_skf(:,s)] = max(squeeze(fSt(:,:,s)),[],2); 
    [~, St_sks(:,s)] = max(squeeze(sSt(:,:,s)),[],2);
end

% Output
fsvar.K = M; fsvar.A = A; fsvar.V = Q; fsvar.H = H; fsvar.R = R;
fsvar.x0 = x0; fsvar.Z = Z; fsvar.pi = pi;
Path.fSt = fSt; Path.sSt = sSt; Path.fxt = fxt; Path.sxt = sxt; 
Path.St_skf = St_skf; Path.St_sks = St_sks; Path.Rs = Rs;
Path.fft = fxt(1:opts.r,:,:); Path.sft = sxt(1:opts.r,:,:);
