%==========================================================================
%  Estimating dynamic connectivity states using factor switching VAR model
%  Resting-state fMRI, Multi-subject analysis
%
%  Authors: Chee-Ming Ting, Universiti Teknologi Malaysia & KAUST (2017)
%==========================================================================
clc; clear all; close all;
cd('C:\Users\0wner\Desktop\f-SVAR-Toolbox');

%---------------------------------------------------------------------%
%                     Resting-state fMRI data                         %
%---------------------------------------------------------------------%
input_dir = 'rs-fmri-data\';
sub_label = {'fmri_rsn_mean_sub05676';'fmri_rsn_mean_sub14864';'fmri_rsn_mean_sub09607'};
[no_subs,~] = size(sub_label);

file = strcat(input_dir,sub_label{1},'.mat');
load(file);
[N, Tr] = size(mean_roi_subs);
T = Tr*no_subs;
Yr = zeros(N,Tr,no_subs); yr = zeros(N,Tr); % Subject-specific signals
Y = zeros(N,T);   % All-subjects signals

for s=1:no_subs
    file = strcat(input_dir,sub_label{s},'.mat');
    load(file); % Load data
    for i = 1:N
        yr(i,:) = (mean_roi_subs(i,:)-mean(mean_roi_subs(i,:)))/std(mean_roi_subs(i,:)); % Normalization
        yr(i,:) = medfilt1(yr(i,:),2); % Filtering
    end   
    Yr(:,:,s) = yr;
    Y(:,(s-1)*Tr+1:(s-1)*Tr+Tr) = yr; % Concatenate data
end

%---------------------------------------------------------------------%
%                     Estimation of f-SVAR Model                      %
%---------------------------------------------------------------------%
kmax = 10; % maximum range of canditate r
[r, ~, bic] = fmselr(Y,kmax); % Select no. of factors by BIC

opts.p = 1;     % VAR model order
opts.K = 3;     % Number of states
opts.r = r;    % Number of factors
opts.wlen = 15; % win length
opts.shift = 1; % win shift
opts.eps = 0.0001; % Stop EM if eps<like improvement
opts.ItrNo = 15; % maximum EM iterations

% Step 1: Initialization based on a factor model
[fsvar0,fm] = fsvarInit(Y,T,opts);

% Step 2: EM estimation of regime-switching factor model
[fsvar,Path,L] = fsvarest(Yr,fsvar0,opts);

% Step 3: Projection to high-dim connectivity states
[varmat] = fsvarproj(Yr,fsvar,Path,fm,opts);

%---------------------------------------------------------------------%
%                           Plot Results                              %
%---------------------------------------------------------------------%
% Resting-state net partitions
nROI = N;
map = dlmread('aal_rsn_map_96ROI.txt');
nNet = 6; % number of networks
nIr = zeros(2,nNet); % Indices of the first and last ROIs of each network
nROI_net = zeros(nNet,1); % Number of ROIs in each network
dEnd = 0; End = 0;
for i = 1:nNet
    ind = find(map(:,3) == i);
    [nROI_net(i,1)] = size(ind,1);
    % Assign indices for the first and last ROI of each network
    nIr(1,i) = End + 1;   nIr(2,i) = nIr(1,i) + nROI_net(i,1) - 1;  End = nIr(2,i);
end
Net_name = {'SCN','AN','SMN','VN','ATN','DMN'};

% Plot estimated state sequence
figure('Name','State sequence','Color',[1 1 1]);
K = opts.K;
for s =1:no_subs
    subplot(no_subs,1,s);
    plot(1:1:Tr,Path.St_sks(:,s),'Color',[0 76 153]/255,'LineWidth',2.3); hold on;
    xlim([1 Tr]); ylim([.75 K+.25]);
    set(gca,'YTick',1:1:K,'fontsize',11);
    if s==no_subs
        xlabel('Time Point', 'fontsize',12); end
    ylabel('States', 'fontsize',12); title(strcat('subject ',num2str(s)));
end

% Plot state VAR connectivity
figure('Name','Coupled f-SVAR Net','Color',[1 1 1],'pos',[100 100 1500 350]);
for j=1:K
subplot(1,K,j);
imagesc(squeeze(varmat.Aco(:,:,j)));
colormap jet; h=colorbar;
caxis([-0.2 0.2]);
for i = 1:nNet-1
    line([nIr(2,i)+0.5 nIr(2,i)+0.5],[0 nIr(2,nNet)+0.5]);  hold on;
    line([0 nIr(2,nNet)+0.5],[nIr(2,i)+0.5 nIr(2,i)+0.5]);  hold on; 
end
set(findobj(gcf,'Type','line'),'Color','k','LineStyle','-','LineWidth',1.5);
ylabel('ROIs','FontSize',14,'fontweight','bold'); xlabel('ROIs','FontSize',14,'fontweight','bold');
set(gca,'XTick',0:10:nROI,'YTick',0:10:nROI);
set(gca,'FontSize',12);  set(h,'fontsize',10); title(strcat('State ',num2str(j)));
end

figure('Name','Decoupled f-SVAR Net','Color',[1 1 1],'pos',[100 100 1500 350]);
for j=1:K
subplot(1,K,j);
imagesc(squeeze(varmat.Ade(:,:,j)));
colormap jet; h=colorbar;
caxis([-0.2 0.2]);
for i = 1:nNet-1
    line([nIr(2,i)+0.5 nIr(2,i)+0.5],[0 nIr(2,nNet)+0.5]);  hold on;
    line([0 nIr(2,nNet)+0.5],[nIr(2,i)+0.5 nIr(2,i)+0.5]);  hold on; 
end
set(findobj(gcf,'Type','line'),'Color','k','LineStyle','-','LineWidth',1.5);
ylabel('ROIs','FontSize',14,'fontweight','bold'); xlabel('ROIs','FontSize',14,'fontweight','bold');
set(gca,'XTick',0:10:nROI,'YTick',0:10:nROI);
set(gca,'FontSize',12);  set(h,'fontsize',10); title(strcat('State ',num2str(j)));
end

figure('Name','Significant Decoupled f-SVAR Net','Color',[1 1 1],'pos',[100 100 1500 350]);
for j=1:K
subplot(1,K,j);
imagesc(squeeze(varmat.Ade_sig(:,:,j)));
colormap jet; h=colorbar;
caxis([-0.2 0.2]);
for i = 1:nNet-1
    line([nIr(2,i)+0.5 nIr(2,i)+0.5],[0 nIr(2,nNet)+0.5]);  hold on;
    line([0 nIr(2,nNet)+0.5],[nIr(2,i)+0.5 nIr(2,i)+0.5]);  hold on; 
end
set(findobj(gcf,'Type','line'),'Color','k','LineStyle','-','LineWidth',1.5);
ylabel('ROIs','FontSize',14,'fontweight','bold'); xlabel('ROIs','FontSize',14,'fontweight','bold');
set(gca,'XTick',0:10:nROI,'YTick',0:10:nROI);
set(gca,'FontSize',12);  set(h,'fontsize',10); title(strcat('State ',num2str(j)));
end

