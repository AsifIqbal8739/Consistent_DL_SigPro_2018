function [coefs,dataC,data] = gererateNoiseAddedSyntheticData(N, m, Dictionary, SNRdB, M)
%-Generates Synthetic Training Data for give SNR-%
%------------------------------------------------------------------
% Courtesy : M. Aharon and M. Elad, Department of Computer Science, 
% Technion, Haifa, Israel
% Modified by Asif Iqbal, University of Melbourne
%------------------------------------------------------------------
% randn('state',sum(100*clock));
% rand('state',sum(100*clock))
if nargin < 5,    M = 1;    end;    % M : # of atoms in a group
    
[dataC,coefs] = CreateDataFromDictionarySimple(Dictionary, N, m, M);

if (SNRdB==0||SNRdB>=80) 
    data=dataC;
    return;
else
    noise = randn(size(dataC));
    actualNoise = calcNoiseFromSNR(SNRdB,dataC, noise);
    data =  dataC + actualNoise;
end

function [D,xOrig] = CreateDataFromDictionarySimple(dictionary, numElements, numCoef, grpSize)
if grpSize == 1
    xOrig = zeros(size(dictionary,2),numElements);

    maxRangeOfCoef = 1;
    coefs = randn(numCoef,numElements)*maxRangeOfCoef;
    xOrig(1:numCoef,:) = coefs;

    for i=1:size(xOrig,2)
        xOrig(:,i) = xOrig(randperm(size(xOrig,1)),i);
    end
    % for i=1:size(xOrig,2)
    %     xOrig(2:end,i) = xOrig(1+randperm(size(xOrig,1)-1),i);
    % end
else
    xOrig = zeros(size(dictionary,2),numElements);
    maxRangeOfCoef = 1;    
    coefs = randn(numCoef*grpSize,numElements)*maxRangeOfCoef;
    
    for i = 1:size(xOrig,2)
        x = randperm(size(xOrig,1)/grpSize,numCoef);
        xx = [2*x-1;2*x];
        xOrig(xx(:),i) = coefs(:,i);
    end
    
    
end
D = dictionary*xOrig;

function  actualNoise = calcNoiseFromSNR(TargerSNR, signal, randomNoise)
signal_2 = sum(signal.^2);
ActualNoise_2 = signal_2./(10^(TargerSNR/10));
noise_2 = sum(randomNoise.^2);
ratio = ActualNoise_2./noise_2;
actualNoise = randomNoise.*repmat(sqrt(ratio),size(randomNoise,1),1);

% function SNR = calcSNR(origSignal, noisySignal)
% errorSignal = origSignal-noisySignal;
% signal_2 = sum(origSignal.^2);
% noise_2 = sum(errorSignal.^2);
% 
% SNRValues = 10*log10(signal_2./noise_2);
% SNR = mean(SNRValues);
