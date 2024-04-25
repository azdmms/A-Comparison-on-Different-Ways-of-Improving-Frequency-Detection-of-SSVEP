%% mway cca azadeh 
clc;
clear all;
close all;

%%
numHarmonics = 1;
N=numHarmonics;
Fs=250;
%% main
%% loading data
% ‘Electrode index’, ‘Time points’, ‘Target index’, and ‘Block index’
load 'C:\Users\DearUser\Downloads\S1.mat\S1.mat';
datapre0=data;
load 'C:\Users\DearUser\Downloads\S2.mat\S2.mat';
datapre1=data;
load 'C:\Users\DearUser\Downloads\S3.mat\S3.mat';
datapre2=data;
load 'C:\Users\DearUser\Downloads\S4.mat\S4.mat';
datapre3=data;
load 'C:\Users\DearUser\Downloads\S5.mat\S5.mat';
datapre4=data;
load 'C:\Users\DearUser\Downloads\S6.mat\S6.mat';
datapre5=data;
load 'C:\Users\DearUser\Downloads\S7.mat\S7.mat';
datapre6=data;
load 'C:\Users\DearUser\Downloads\S8.mat\S8.mat';
datapre7=data;
load 'C:\Users\DearUser\Downloads\S9.mat\S9.mat';
datapre8=data;
load 'C:\Users\DearUser\Downloads\S10.mat\S10.mat';
datapre9=data;
load 'C:\Users\DearUser\Downloads\S11.mat\S11.mat';
datapre10=data;
load 'C:\Users\DearUser\Downloads\S12.mat\S12.mat';
datapre11=data;
load 'C:\Users\DearUser\Downloads\S13.mat\S13.mat';
datapre12=data;
load 'C:\Users\DearUser\Downloads\S14.mat\S14.mat';
datapre13=data;
load 'C:\Users\DearUser\Downloads\S15.mat\S15.mat';
datapre14=data;
load 'C:\Users\DearUser\Downloads\S16.mat\S16.mat';
datapre15=data;
load 'C:\Users\DearUser\Downloads\S17.mat\S17.mat';
datapre16=data;
load 'C:\Users\DearUser\Downloads\S18.mat\S18.mat';
datapre17=data;
load 'C:\Users\DearUser\Downloads\S19.mat\S19.mat';
datapre18=data;
load 'C:\Users\DearUser\Downloads\S20.mat\S20.mat';
datapre19=data;
load 'C:\Users\DearUser\Downloads\S21.mat\S21.mat';
datapre20=data;
load 'C:\Users\DearUser\Downloads\S22.mat\S22.mat';
datapre21=data;
load 'C:\Users\DearUser\Downloads\S23.mat\S23.mat';
datapre22=data;
load 'C:\Users\DearUser\Downloads\S24.mat\S24.mat';
datapre23=data;
load 'C:\Users\DearUser\Downloads\S25.mat\S25.mat';
datapre24=data;
load 'C:\Users\DearUser\Downloads\S26.mat\S26.mat';
datapre25=data;
load 'C:\Users\DearUser\Downloads\S27.mat\S27.mat';
datapre26=data;
load 'C:\Users\DearUser\Downloads\S28.mat\S28.mat';
datapre27=data;
load 'C:\Users\DearUser\Downloads\S29.mat\S29.mat';
datapre28=data;
load 'C:\Users\DearUser\Downloads\S30.mat\S30.mat';
datapre29=data;
load 'C:\Users\DearUser\Downloads\S31.mat\S31.mat';
datapre30=data;
load 'C:\Users\DearUser\Downloads\S32.mat\S32.mat';
datapre31=data;
load 'C:\Users\DearUser\Downloads\S33.mat\S33.mat';
datapre32=data;
load 'C:\Users\DearUser\Downloads\S34.mat\S34.mat';
datapre33=data;
load 'C:\Users\DearUser\Downloads\S35.mat\S35.mat';
datapre34=data;
datapre=datapre0;
%% combining data
numberofsubj=4;
dataT(1,:,:,:,:)=datapre0;
dataT(2,:,:,:,:)=datapre1;
dataT(3,:,:,:,:)=datapre2;
dataT(4,:,:,:,:)=datapre3;
dataT(5,:,:,:,:)=datapre4;
dataT(6,:,:,:,:)=datapre5;
dataT(7,:,:,:,:)=datapre6;
dataT(8,:,:,:,:)=datapre7;
dataT(9,:,:,:,:)=datapre8;
dataT(10,:,:,:,:)=datapre9;
dataT(11,:,:,:,:)=datapre10;
dataT(12,:,:,:,:)=datapre11;
dataT(13,:,:,:,:)=datapre12;
dataT(14,:,:,:,:)=datapre13;
dataT(15,:,:,:,:)=datapre14;
dataT(16,:,:,:,:)=datapre15;
dataT(17,:,:,:,:)=datapre16;
dataT(18,:,:,:,:)=datapre17;
dataT(19,:,:,:,:)=datapre18;
dataT(20,:,:,:,:)=datapre19;
dataT(21,:,:,:,:)=datapre20;
dataT(22,:,:,:,:)=datapre21;
dataT(23,:,:,:,:)=datapre22;
dataT(24,:,:,:,:)=datapre23;
dataT(25,:,:,:,:)=datapre24;
dataT(26,:,:,:,:)=datapre25;
dataT(27,:,:,:,:)=datapre26;
dataT(28,:,:,:,:)=datapre27;
dataT(29,:,:,:,:)=datapre28;
dataT(30,:,:,:,:)=datapre29;
dataT(31,:,:,:,:)=datapre30;
dataT(32,:,:,:,:)=datapre31;
dataT(33,:,:,:,:)=datapre32;
dataT(34,:,:,:,:)=datapre33;
dataT(35,:,:,:,:)=datapre34;
%% jodasazi zaman haye bein task ha
%data=zeros(64,1375,40,6);
for k=121:1:1375
    data0(:,k-120,:,:)=datapre(:,k,:,:);
end
for k=121:1:1375
    data1(:,k-120,:,:)=datapre1(:,k,:,:);
end
for k=121:1:1375
    data2(:,k-120,:,:)=datapre2(:,k,:,:);
end
for k=121:1:1375
    data3(:,k-120,:,:)=datapre3(:,k,:,:);
end
for k=121:1:1375
    data4(:,k-120,:,:)=datapre4(:,k,:,:);
end
for k=121:1:1375
    data5(:,k-120,:,:)=datapre5(:,k,:,:);
end
for k=121:1:1375
    data6(:,k-120,:,:)=datapre6(:,k,:,:);
end
for k=121:1:1375
    data7(:,k-120,:,:)=datapre7(:,k,:,:);
end
for k=121:1:1375
    data8(:,k-120,:,:)=datapre8(:,k,:,:);
end
for k=121:1:1375
    data9(:,k-120,:,:)=datapre9(:,k,:,:);
end
for k=121:1:1375
    data10(:,k-120,:,:)=datapre10(:,k,:,:);
end
for k=121:1:1375
    data11(:,k-120,:,:)=datapre11(:,k,:,:);
end
for k=121:1:1375
    data12(:,k-120,:,:)=datapre12(:,k,:,:);
end
for k=121:1:1375
    data13(:,k-120,:,:)=datapre13(:,k,:,:);
end
for k=121:1:1375
    data14(:,k-120,:,:)=datapre14(:,k,:,:);
end
for k=121:1:1375
    data15(:,k-120,:,:)=datapre15(:,k,:,:);
end
for k=121:1:1375
    data16(:,k-120,:,:)=datapre16(:,k,:,:);
end
for k=121:1:1375
    data17(:,k-120,:,:)=datapre17(:,k,:,:);
end
for k=121:1:1375
    data18(:,k-120,:,:)=datapre18(:,k,:,:);
end
for k=121:1:1375
    data19(:,k-120,:,:)=datapre19(:,k,:,:);
end
for k=121:1:1375
    data20(:,k-120,:,:)=datapre20(:,k,:,:);
end
for k=121:1:1375
    data21(:,k-120,:,:)=datapre21(:,k,:,:);
end
for k=121:1:1375
    data22(:,k-120,:,:)=datapre22(:,k,:,:);
end
for k=121:1:1375
    data23(:,k-120,:,:)=datapre23(:,k,:,:);
end
for k=121:1:1375
    data24(:,k-120,:,:)=datapre24(:,k,:,:);
end
for k=121:1:1375
    data25(:,k-120,:,:)=datapre25(:,k,:,:);
end
for k=121:1:1375
    data26(:,k-120,:,:)=datapre26(:,k,:,:);
end
for k=121:1:1375
    data27(:,k-120,:,:)=datapre27(:,k,:,:);
end
for k=121:1:1375
    data28(:,k-120,:,:)=datapre28(:,k,:,:);
end
for k=121:1:1375
    data29(:,k-120,:,:)=datapre29(:,k,:,:);
end
for k=121:1:1375
    data30(:,k-120,:,:)=datapre30(:,k,:,:);
end
for k=121:1:1375
    data31(:,k-120,:,:)=datapre31(:,k,:,:);
end
for k=121:1:1375
    data32(:,k-120,:,:)=datapre32(:,k,:,:);
end
for k=121:1:1375
    data33(:,k-120,:,:)=datapre33(:,k,:,:);
end
for k=121:1:1375
    data34(:,k-120,:,:)=datapre34(:,k,:,:);
end
for k=121:1:1375
    data35(:,k-120,:,:)=datapre35(:,k,:,:);
end

%% jodasazi frequency va electrode haye matlub maghale
%dar in maghale 8 frequency estefade shode and vali dar dade haye mojud 40
%ta vali ma 8 taye an ra entekhab mikonim
fr=(8:0.2:15.8) ;
freq=[8,9.2,11,12,13,13.4,15,15.8];
i=[46,44,48,50,52,51,62,63];
countelect=0;
countfreq=0;
c=0;
d=0;
[sharedvals,idx] = intersect(fr,freq,'stable');
% frequency haye matlub maghale joda shodand va dakhele Xfreq data un freq
% ha rikhte shod
for i=idx
    Xfreq(:,:,:,:)=data0(:,:,i,:);
end
% electrode haye matlub maghale joda shodand va dakhele X_elect data un
% electrode
% ha rikhte shod
for j=i
    X_elect(:,:,:,:)=Xfreq(j,:,:,:);
end

for i=idx
    Xfreq1(:,:,:,:)=data1(:,:,i,:);
end
for j=i
    X_elect1(:,:,:,:)=Xfreq1(j,:,:,:);
end

for i=idx
    Xfreq2(:,:,:,:)=data2(:,:,i,:);
end
for j=i
    X_elect2(:,:,:,:)=Xfreq2(j,:,:,:);
end
for i=idx
    Xfreq3(:,:,:,:)=data3(:,:,i,:);
end
for j=i
    X_elect3(:,:,:,:)=Xfreq3(j,:,:,:);
end
for i=idx
    Xfreq4(:,:,:,:)=data4(:,:,i,:);
end
for j=i
    X_elect4(:,:,:,:)=Xfreq4(j,:,:,:);
end
for i=idx
    Xfreq5(:,:,:,:)=data5(:,:,i,:);
end
for j=i
    X_elect5(:,:,:,:)=Xfreq5(j,:,:,:);
end
for i=idx
    Xfreq6(:,:,:,:)=data6(:,:,i,:);
end
for j=i
    X_elect6(:,:,:,:)=Xfreq6(j,:,:,:);
end
for i=idx
    Xfreq7(:,:,:,:)=data7(:,:,i,:);
end
for j=i
    X_elect7(:,:,:,:)=Xfreq7(j,:,:,:);
end
for i=idx
    Xfreq8(:,:,:,:)=data8(:,:,i,:);
end
for j=i
    X_elect8(:,:,:,:)=Xfreq8(j,:,:,:);
end
for i=idx
    Xfreq9(:,:,:,:)=data9(:,:,i,:);
end
for j=i
    X_elect9(:,:,:,:)=Xfreq9(j,:,:,:);
end
for i=idx
    Xfreq10(:,:,:,:)=data10(:,:,i,:);
end
for j=i
    X_elect10(:,:,:,:)=Xfreq10(j,:,:,:);
end
for i=idx
    Xfreq11(:,:,:,:)=data11(:,:,i,:);
end
for j=i
    X_elect11(:,:,:,:)=Xfreq11(j,:,:,:);
end
for i=idx
    Xfreq12(:,:,:,:)=data12(:,:,i,:);
end
for j=i
    X_elect12(:,:,:,:)=Xfreq12(j,:,:,:);
end
for i=idx
    Xfreq13(:,:,:,:)=data13(:,:,i,:);
end
for j=i
    X_elect13(:,:,:,:)=Xfreq13(j,:,:,:);
end
for i=idx
    Xfreq14(:,:,:,:)=data14(:,:,i,:);
end
for j=i
    X_elect14(:,:,:,:)=Xfreq14(j,:,:,:);
end
for i=idx
    Xfreq15(:,:,:,:)=data15(:,:,i,:);
end
for j=i
    X_elect15(:,:,:,:)=Xfreq15(j,:,:,:);
end
for i=idx
    Xfreq16(:,:,:,:)=data16(:,:,i,:);
end
for j=i
    X_elect16(:,:,:,:)=Xfreq16(j,:,:,:);
end
for i=idx
    Xfreq17(:,:,:,:)=data17(:,:,i,:);
end
for j=i
    X_elect17(:,:,:,:)=Xfreq17(j,:,:,:);
end
for i=idx
    Xfreq18(:,:,:,:)=data18(:,:,i,:);
end
for j=i
    X_elect18(:,:,:,:)=Xfreq18(j,:,:,:);
end
for i=idx
    Xfreq19(:,:,:,:)=data19(:,:,i,:);
end
for j=i
    X_elect19(:,:,:,:)=Xfreq19(j,:,:,:);
end
for i=idx
    Xfreq20(:,:,:,:)=data20(:,:,i,:);
end
for j=i
    X_elect20(:,:,:,:)=Xfreq20(j,:,:,:);
end
for i=idx
    Xfreq21(:,:,:,:)=data21(:,:,i,:);
end
for j=i
    X_elect21(:,:,:,:)=Xfreq21(j,:,:,:);
end
for i=idx
    Xfreq22(:,:,:,:)=data22(:,:,i,:);
end
for j=i
    X_elect22(:,:,:,:)=Xfreq22(j,:,:,:);
end
for i=idx
    Xfreq23(:,:,:,:)=data23(:,:,i,:);
end
for j=i
    X_elect23(:,:,:,:)=Xfreq23(j,:,:,:);
end
for i=idx
    Xfreq24(:,:,:,:)=data24(:,:,i,:);
end
for j=i
    X_elect24(:,:,:,:)=Xfreq24(j,:,:,:);
end
for i=idx
    Xfreq25(:,:,:,:)=data25(:,:,i,:);
end
for j=i
    X_elect25(:,:,:,:)=Xfreq25(j,:,:,:);
end
for i=idx
    Xfreq26(:,:,:,:)=data26(:,:,i,:);
end
for j=i
    X_elect26(:,:,:,:)=Xfreq26(j,:,:,:);
end
for i=idx
    Xfreq27(:,:,:,:)=data27(:,:,i,:);
end
for j=i
    X_elect27(:,:,:,:)=Xfreq27(j,:,:,:);
end
for i=idx
    Xfreq28(:,:,:,:)=data28(:,:,i,:);
end
for j=i
    X_elect28(:,:,:,:)=Xfreq28(j,:,:,:);
end
for i=idx
    Xfreq29(:,:,:,:)=data29(:,:,i,:);
end
for j=i
    X_elect29(:,:,:,:)=Xfreq29(j,:,:,:);
end
for i=idx
    Xfreq30(:,:,:,:)=data30(:,:,i,:);
end
for j=i
    X_elect30(:,:,:,:)=Xfreq30(j,:,:,:);
end
for i=idx
    Xfreq31(:,:,:,:)=data31(:,:,i,:);
end
for j=i
    X_elect31(:,:,:,:)=Xfreq31(j,:,:,:);
end
for i=idx
    Xfreq32(:,:,:,:)=data32(:,:,i,:);
end
for j=i
    X_elect32(:,:,:,:)=Xfreq32(j,:,:,:);
end
for i=idx
    Xfreq33(:,:,:,:)=data33(:,:,i,:);
end
for j=i
    X_elect33(:,:,:,:)=Xfreq33(j,:,:,:);
end
for i=idx
    Xfreq34(:,:,:,:)=data34(:,:,i,:);
end
for j=i
    X_elect34(:,:,:,:)=Xfreq34(j,:,:,:);
end
for i=idx
    Xfreq35(:,:,:,:)=data35(:,:,i,:);
end
for j=i
    X_elect35(:,:,:,:)=Xfreq35(j,:,:,:);
end
%% Load data
load 'C:\Users\DearUser\SSVEPdata';
% 8 channels x 1000 points x 20 trials x 8 stimulus frequencies
eegData=SSVEPdata3;
%%
numChannels = size(eegData, 1);      
numTimePoints = size(eegData, 2);     
numTrials = size(eegData, 3);         
numHarmonics = 1; 
numFreqs=8;
Fs=250;
t_length=4; 
n_correct_freq=zeros(1,8);
stimulusFreqs=[9.75 8.75 7.75 5.75 8 9.2 11 12];
sti_f=stimulusFreqs;
%original reference signal
N=numHarmonics;    % number of harmonics
ref(1,:,:)=refsig(sti_f(1),Fs,t_length*Fs,N);
ref(2,:,:)=refsig(sti_f(2),Fs,t_length*Fs,N);
ref(3,:,:)=refsig(sti_f(3),Fs,t_length*Fs,N);
ref(4,:,:)=refsig(sti_f(4),Fs,t_length*Fs,N);
ref(5,:,:)=refsig(sti_f(5),Fs,t_length*Fs,N);
ref(6,:,:)=refsig(sti_f(6),Fs,t_length*Fs,N);
ref(7,:,:)=refsig(sti_f(7),Fs,t_length*Fs,N);
ref(8,:,:)=refsig(sti_f(8),Fs,t_length*Fs,N);
% originalReferenceSignals = zeros(numHarmonics * 2, numTimePoints);
% for h = 1:numHarmonics
%     originalReferenceSignals(h, :) = sin(2 * pi * stimulusFreqs * h * (1:numTimePoints) / samplingRate);
%     originalReferenceSignals(h + numHarmonics, :) = cos(2 * pi * stimulusFreqs * h * (1:numTimePoints) / samplingRate);
% end
optimalReferenceSignals = zeros(1, numTimePoints);

%%
for m = 1:numFreqs
    %Xm
    Xm = eegData(:, :, :, m);
    %bm
    bm = zeros(numChannels, 1);
    while true
        %Fix w3
        w3=rand(24,1);
        w1 = zeros(numChannels, 1);
        v = zeros(numHarmonics * 2, 1);
        
%         for k = 1:numTrials
            Y1 = squeeze(ref(m,:,:));
            X1 = tensorprod(Xm(:, :, :), w3);
            
            [W1, v, r1] = cca(X1,Y1)
            %w1 = w1 + X1' * Y1 * inv(Y1' * Y1);
            %v = v + inv(Y1' * Y1) * Y1' * X1;
%         end
        
       % w1 = w1 / numTrials;
       % v = v / numTrials;
        
        %Fix w1 and v
        
        w3 = zeros(numTrials, 1);
        
       % for k = 1:numTrials
            X3 = tensorprod(Xm(:, :, :), w1);
            Y3 = tensorprod(squeeze(ref(m,:,:)),v);
            [W3, nn, r1] = cca(X3,Y3)
            %w3(k) = X3' * Y3 * inv(Y3' * Y3);
        %end
        if norm(w1 - prevW1) < epsilon && norm(w3 - prevW3) < epsilon
            break;
        end
        
        prevW1 = w1;
        prevW3 = w3;
    end
    
    %zm
    zm = tensorprod(Xm(:, :, :) ,w1);
    zm=tensorprod(zm,w3);
    optimalReferenceSignals = optimalReferenceSignals + zm';
   
end

%Normalize
optimalReferenceSignals = optimalReferenceSignals / numFreqs;





classificationAccuracy = zeros(numTrials, 1);
for k = 1:numTrials
    Xtest = squeeze(Xm(:, :, k));
    b_hat = inv(Xtest' * Xtest) * Xtest' * optimalReferenceSignals';
    z_hat = optimalReferenceSignals' * b_hat;
    
    correlationCoefficients = 1 - norm(optimalReferenceSignals' - z_hat).^2 ./ norm(optimalReferenceSignals').^2;
    [~, ftarget] = max(correlationCoefficients);
    
    if ftarget == m
        classificationAccuracy(k) = 1;
    end
    for jj = 1:numFreqs
         if ftarget==numFreqs
            n_correct_freq(jj)=n_correct_freq(jj)+1;
         end
    end
end
averageClassificationAccuracy = mean(classificationAccuracy);



%% Plot accuracy
% accuracy=100*n_correct/n_sti/n_run;
accuracy_per_freq=100*n_correct_freq/n_run;
col={'b-*','r-o'};
plot(TW,averageClassificationAccuracy,col{1},'LineWidth',1);
hold on;
xlabel('Time window length (s)');
ylabel('Accuracy (%)');
grid;
xlim([0.75 5.25]);
ylim([0 100]);
set(gca,'xtick',1:5,'xticklabel',1:5);
title('\bf MsetCCA');
h=legend({'CCA','MsetCCA'});
set(h,'Location','SouthEast');
sim=accuracy_per_freq;
figure;
ylim([0 100]);
freq=[8,9.2,11,12,13,13.4,15,15.8];
plot(freq,sim,'r-o');
ylim([0 100]);
title('accuracy for each stimulus frequency');
xlabel('freq(Hz)');
ylabel('Accuracy (%)');




%% function declaration
%% cca
function [Wx, Wy, r] = cca(X,Y)

% CCA calculate canonical correlations
%
% [Wx Wy r] = cca(X,Y) where Wx and Wy contains the canonical correlation
% vectors as columns and r is a vector with corresponding canonical
% correlations. The correlations are sorted in descending order. X and Y
% are matrices where each column is a sample. Hence, X and Y must have
% the same number of columns.
%
% Example: If X is M*K and Y is N*K there are L=MIN(M,N) solutions. Wx is
% then M*L, Wy is N*L and r is L*1.
%
%
% ?? 2000 Magnus Borga, Linköpings universitet

% --- Calculate covariance matrices ---

z = [X;Y];
C = cov(z.');
sx = size(X,1);
sy = size(Y,1);
Cxx = C(1:sx, 1:sx) + 10^(-8)*eye(sx);
Cxy = C(1:sx, sx+1:sx+sy);
Cyx = Cxy';
Cyy = C(sx+1:sx+sy, sx+1:sx+sy) + 10^(-8)*eye(sy);
invCyy = inv(Cyy);

% --- Calcualte Wx and r ---

[Wx,r] = eig(inv(Cxx)*Cxy*invCyy*Cyx); % Basis in X
r = sqrt(real(r));      % Canonical correlations

% --- Sort correlations ---

V = fliplr(Wx);		% reverse order of eigenvectors
r = flipud(diag(r));	% extract eigenvalues and reverse their order
[r,I]= sort((real(r)));	% sort reversed eigenvalues in ascending order
r = flipud(r);		% restore sorted eigenvalues into descending order
for j = 1:length(I)
  Wx(:,j) = V(:,I(j));  % sort reversed eigenvectors in ascending order
end
Wx = fliplr(Wx);	% restore sorted eigenvectors into descending order

% --- Calcualte Wy  ---

Wy = invCyy*Cyx*Wx;     % Basis in Y
% Wy = Wy./repmat(sqrt(sum(abs(Wy).^2)),sy,1); % Normalize Wy
end
%% refsig
function y=refsig(f, S, T, N)

% f-- the fundemental frequency
% S-- the sampling rate
% T-- the number of sampling points
% N-- the number of harmonics


for i=1:N
   for j=1:T
    t= j/S;
    y(2*i-1,j)=sin(2*pi*(i*f)*t);
    y(2*i,j)=cos(2*pi*(i*f)*t);
   end
end
end


