% Dictionary recovery simulation scenario
clc; clear all; close all;

%% Data Stuff
% rng('default');
m = 25;     n = 50;     N = 2500;       % D(m,n), Y(m,N) 
K = 3;      % The sparsity parameter
SnRdB = 10;
noIt = 11*K^2;   
if noIt > 100;  noIt = 100; end;

Dict_O = normc(randn(m,n));     % Generating Dictionary
[~,~,Yn] = gererateNoiseAddedSyntheticData(N,K,Dict_O,SnRdB);   % Noisy Signals
Dict = normc(Yn(:,randperm(size(Yn,2),n))); % Initial Dictionary

Methods = {'KSVD','S1','A1','A2'};
[Count_KSVD,Count_S1,Count_A1,Count_A2] = deal(zeros(1,noIt));

%% Learning the dictionaries
% OMP needed for KSVD and S1
% Count_KSVD = DictLearn(Yn,Dict,Dict_O,noIt,K,Methods{1},0);
% alpha = 0.2;
% Count_S1 = DictLearn(Yn,Dict,Dict_O,noIt,K,Methods{2},alpha);
alpha = 0.2;
Count_A1 = DictLearn(Yn,Dict,Dict_O,noIt,K,Methods{3},alpha);
Count_A2 = DictLearn(Yn,Dict,Dict_O,noIt,K,Methods{4},alpha);

figure;
plot(Count_KSVD,'r--','LineWidth',2); hold on;
plot(Count_S1,'b-.','LineWidth',2);
plot(Count_A1,'k-','LineWidth',2);
plot(Count_A2,'m:','LineWidth',2);

xlabel('Iterations');   ylabel('Atom Recovery Percentage');
title(sprintf('Dictionary Recovery for SNR: %d dB',SnRdB));
legend(Methods,'Location','SE','FontSize',13);
