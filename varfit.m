function [PI,Q] = varfit(p,Y)
%==========================================================================
%  Fitting a m-dimensional VAR(p) on multivariate time series
% 
%  Inputs: p - model order, Y - m*T time series data
%  Outputs: PI - m x mp AR coefficient matrix
%           Q  - m x m covariance matrix
%
%  Author: Chee-Ming Ting, , Universiti Teknologi Malaysia & KAUST
%==========================================================================

    %----------------------------------------------------------------------%
    % MLE of VAR(p) - Ordinary least squares (OLS) estimation per equation %
    %----------------------------------------------------------------------%
    [m, n]=size(Y); % m - Dimension of time series n - Number of sample points
    PI = zeros(m,m*p); % Block matrix of VAR coefficients
    xt = zeros(m*p,n); % Store previous p observations
    e  = zeros(m,n-p);
    
    % Calculate denominator constant, independently of j
    PI_denorm = 0;
    for t=p+1:n
        temp = fliplr(Y(:,t-p:t-1));
        xt(:,t) = temp(:); 
        clear temp;
        PI_denorm = PI_denorm + xt(:,t)*xt(:,t)';
    end
%     PI_denorm = inv(PI_denorm);
     PI_denorm = pinv(PI_denorm);
    
    % Estimation of the coefficients of the jth equation of VAR
    for j=1:m
        % Calculate norminator term
        PI_norm = 0;     
        for t=p+1:n   
            PI_norm = PI_norm + Y(j,t)*xt(:,t)';       
        end
        PI(j,:)   = PI_norm * PI_denorm;
    end
    
    % Estimate residuals
    for t=p+1:n
        e(:,t) = Y(:,t) - PI * xt(:,t);
    end
    
    % Estimate noise covariance matrix from residuals
    Q = cov(e(:,p+1:end)');

end

