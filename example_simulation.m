%==========================================================================
%  Estimating dynamic connectivity states using factor switching VAR model
%  Simulation studies - 2-state switching VAR(1) with block-diagonal structure
%
%  Authors: Chee-Ming Ting, Universiti Teknologi Malaysia & KAUST (2017)
%==========================================================================
clc; clear all; close all;
cd('C:\Users\0wner\Desktop\f-SVAR-Toolbox');

%-------------------------------------------------------------------------%
%                     Simulation of SVAR(1)                               %
%-------------------------------------------------------------------------%
K  = 2; % no. of states
p0 = 1; % True model order
nROI = 3;       % no. of ROIs
dim_roi = 10;   % Dimension of voxel time-series in each ROI
N = nROI*dim_roi; % Dimension of total time series
T  = 400; % Total no of time points
Tb = round(T/(2*K)); % No of time points per regime

% Synthesize block-diagonal VAR matrix in each state
Alag = zeros(N,N,p0,K);
A    = zeros(N*p0,N*p0,K);
Ua = [0.4 0.6]; % for K=2, state 1: aij~(-0.4 0.4)  state 2: aij~(-0.6 0.6)
for j=1:K
    while 1
    yI = zeros(2,nROI); % Indices of first & last voxel in each ROI
    lam = 2;
    dEnd = 0;
    Lo = -Ua(j); Up = Ua(j); % strenght of connections
    for i = 1:nROI
        yI(1,i) = dEnd + 1;   yI(2,i) = yI(1,i) + dim_roi - 1;  dEnd = yI(2,i);
        for k=1:p0
            Alag(yI(1,i):yI(2,i),yI(1,i):yI(2,i),k,j) = (Lo + (Up-Lo)*rand(dim_roi,dim_roi))./ lam^(k-1); end        
    end    
    % Check stability of VAR
    for k=1:p0 
        A(1:N,(k-1)*N+1:(k-1)*N+N,j) =  Alag(:,:,k,j);
        if k<p0
            A(k*N+1:k*N+N,(k-1)*N+1:(k-1)*N+N,j) = eye(N,N); end
    end
    if (max(abs(eig(squeeze(A(:,:,j))))) < 1)
         disp('VAR is stationary');
        break
    end
    end
end

V   = eye(N); % Noise cov
for j=1:K
    A_cell = cell(1,p0);
    for k=1:p0 
        A_cell(1,k) = {Alag(:,:,k,j)}; end
    Spec{1,j} = vgxset('AR',A_cell,'Q',V);
end

% Generate time series
Y = zeros(N,T);
St_true = zeros(1,T);
for j=1:K
    Y(:,(j-1)*Tb+1:(j-1)*Tb+Tb) = vgxsim(Spec{1,j},Tb)';
    Y(:,(T/2)+(j-1)*Tb+1:(T/2)+(j-1)*Tb+Tb) = vgxsim(Spec{1,j},Tb)';
    St_true(1,(j-1)*Tb+1:(j-1)*Tb+Tb) = j;
    St_true(1,(T/2)+(j-1)*Tb+1:(T/2)+(j-1)*Tb+Tb) = j;
end
clear Spec

%-------------------------------------------------------------------------%
%                           factor-SVAR                                   %
%-------------------------------------------------------------------------%
Yr(:,:,1) = Y(:,1:T); % one replicate

opts.p = p0;     % VAR model order
opts.K = K;     % Number of states
opts.r = 20;     % Number of factors
opts.wlen = 15; % win length
opts.shift = 1; % win shift
opts.eps = 0.0001; % Stop EM if eps<like improvement
opts.ItrNo = 10; % maximum EM iterations

% Step 1: Initialization based on a factor model
[fsvar0,fm] = fsvarInit(Y(:,1:T),T,opts);

% Step 2: EM estimation of regime-switching factor model
[fsvar,Path,L] = fsvarest(Yr,fsvar0,opts);

% Step 3: Projection to high-dim connectivity states
[varmat] = fsvarproj(Yr,fsvar,Path,fm,opts);


% Plot estimated state sequence
figure('Name','State sequence','Color',[1 1 1]);
subplot(2,1,1);
plot(1:1:T,St_true(:),'Color',[153 0 0]/255,'LineWidth',2.3);
ylim([.75 K+.25]); ylabel('States', 'fontsize',12); title('True state sequence','fontsize',12);
subplot(2,1,2);
plot(1:1:T,Path.St_sks(:,1),'Color',[0 76 153]/255,'LineWidth',2.3);
xlim([1 T]); ylim([.75 K+.25]);
set(gca,'YTick',1:1:K,'fontsize',11);
xlabel('Time Point', 'fontsize',12); ylabel('States', 'fontsize',12);
title('Estimated state sequence','fontsize',12);

% Plot estimated state connectivity matrices
figure('Name','f-SVAR Net','Color',[1 1 1],'pos',[300 100 900 700]);
for j=1:K
% True VAR coeff matrix in each state
subplot(2,K,j);
imagesc(squeeze(A(1:N,:,j)));
colormap hot; h=colorbar;
ylabel('ROIs','FontSize',14,'fontweight','bold'); xlabel('ROIs','FontSize',14,'fontweight','bold');
set(gca,'XTick',0:10:N,'YTick',0:10:N);
set(gca,'FontSize',12);  set(h,'fontsize',10); title(strcat('True state ',num2str(j)),'fontsize',12);

% Decoupled f-SVAR estimator (significant coefficients)
subplot(2,K,K+j);
imagesc(squeeze(varmat.Ade_sig(:,:,j)));
colormap hot; h=colorbar;
ylabel('ROIs','FontSize',14,'fontweight','bold'); xlabel('ROIs','FontSize',14,'fontweight','bold');
set(gca,'XTick',0:10:N,'YTick',0:10:N);
set(gca,'FontSize',12);  set(h,'fontsize',10); title(strcat('Estimated state ',num2str(j)),'fontsize',12);
end

