function [varmat] = fsvarproj(Yr,fsvar,Path,fm,opts)
%--------------------------------------------------------------------------
% Title    : Estimation of High-dimensional VAR connectivity states
% Function : Project the state-specific VAR coeff matrix from factor space
%            to observation space. Compute two f-SVAR estimators
%            (1) Coupled  (common factor loadings)
%            (2) Decoupled (state-dependent factor loadings)
%
% Inputs:  fsvar - Struct with EM-estimated f-SVAR parameters
%          Path  - Struct with estimated regimes by switching Kalman filter
%          fm    - Struct with PCA-estimated factor model
% 
% Outputs: varmat: Structure with projected state connectivity matrix
%               Ade(:,:,j) - Estimated Coupled SVAR matrix (NxNp) at state j
%               Ade_sig(:,:,j) - Ade with significant coefficients
%               Aco(:,:,j) - Estimated Decoupled SVAR matrix (NxNp)at state j
% 
% Authors: Chee-Ming Ting, Universiti Teknologi Malaysia & KAUST (2017)
%          Siti Balqis Samdin, Universiti Teknologi Malaysia
%
% Reference:
%       C.-M. Ting, H. Ombao, S. B. Samdin and S.-H Salleh,
%       “Estimating dynamic connectivity States in fMRI using 
%        regime-swtiching factor model,” IEEE Trans. Medical Imaging. 2017
%--------------------------------------------------------------------------
[N,Tr,Rs] = size(Yr);
T = Tr*Rs;
K = opts.K;
r = opts.r;
p = opts.p;

% Pooling data of each regime
ySt = zeros(N,T,K);
tj = zeros(K,1);
for j=1:K
    t=1;
    for s=1:Rs
        for i=1:Tr
            if Path.St_sks(i,s) == j
               ySt(:,t,j)=Yr(:,i,s);  t=t+1;
            end
        end
    end
    tj(j) = t-1;
end

% Decoupled factor-SVAR with regime-dependet Q: Fit stationary factor-VAR to each regime
Qde = zeros(N,r,K); ARfde = zeros(r,r*p,K); Vde = zeros(r,r,K); covfde_j = zeros(r,r,K);
for j=1:K
    [Qde(:,:,j),covfde,fde,~,~] = fmest(squeeze(ySt(:,1:tj(j),j)),r);
    [ARfde(:,:,j),Vde(:,:,j)] = varfit(p,fde);
    covfde_j(:,:,j)= diag(diag(covfde));
    clear fde covfde 
end
clear ySt
varmat.Ade = zeros(N,N*p,K);
for j=1:K
    for k=1:p
        varmat.Ade(:,(k-1)*N+1:(k-1)*N+N,j) = Qde(:,:,j) * ARfde(:,(k-1)*r+1:(k-1)*r+r,j) *  Qde(:,:,j)';
    end
end

% Coupled factor-SVAR with regime-common Q
varmat.Aco = zeros(N,N*p,K);
for j=1:K
    for k=1:p
        varmat.Aco(:,(k-1)*N+1:(k-1)*N+N,j) = fm.Q * fsvar.A(1:r,(k-1)*r+1:(k-1)*r+r,j) * fm.Q';
    end
end

%-------------------------------------------------------------------------%
%      Significance Test of connections based on asymtotic normality      %
%-------------------------------------------------------------------------%
varmat.Ade_sig = zeros(N,N*p,K); % Decoupled f-SVAR with significant coeff
varmat.Ade_sig(:,:,:) = varmat.Ade(:,:,:);
for k=1:K
    G = kron((Qde(:,:,k)*Vde(:,:,k)*Qde(:,:,k)'),(Qde(:,:,k)*inv(covfde_j(:,:,k))*Qde(:,:,k)'));
    for i=1:N
        for j=1:N
            di = N*(i-1) + j;
            ts = varmat.Ade_sig(j,i,k)/sqrt(G(di,di)/tj(k));
            [h(j,i),pval(j,i)] = ztest(ts,0,1,0.05/(N*N));
            if h(j,i) == 0
               varmat.Ade_sig(j,i,k) = 0;
            end
        end
    end
    clear G
end

end

