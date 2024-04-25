%% mset cca azadeh 
clc;
clear all;
close all;


%% 
Fs=250;                                  % sampling rate
t_length=4;                              % data length (4 s)
TW=1:1:t_length;% time window haye mokhtalefi ke mikhaym
TW_p=round(TW*Fs);
n_run=24;                                % number of used runs
sti_f1=[8,9.2,11,12,13,13.4,15,15.8];             % stimulus frequencies
sti_f=[9.75 8.75 7.75 5.75 8 9.2 11 12]; 
n_sti=length(sti_f);                     % number of stimulus frequencies
n_correct=zeros(2,length(TW));
n_correct_freq=zeros(1,8);


%% Load data
load 'C:\Users\DearUser\SSVEPdata';
% 8 channels x 1000 points x 20 trials x 8 stimulus frequencies


%% normal CCA 
% normal references
N=2;    % number of harmonics
ref1=refsig(sti_f(1),Fs,t_length*Fs,N);
ref2=refsig(sti_f(2),Fs,t_length*Fs,N);
ref3=refsig(sti_f(3),Fs,t_length*Fs,N);
ref4=refsig(sti_f(4),Fs,t_length*Fs,N);
ref5=refsig(sti_f(5),Fs,t_length*Fs,N);
ref6=refsig(sti_f(6),Fs,t_length*Fs,N);
ref7=refsig(sti_f(7),Fs,t_length*Fs,N);
ref8=refsig(sti_f(8),Fs,t_length*Fs,N);
% Recognition
for run=1:24
    for tw_length=1:4       % time window length:  1s:1s:4s
        fprintf('CCA Processing... TW %fs, No.crossvalidation %d \n',TW(tw_length),run);
        for j=1:8
            [wx1,wy1,r1]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),ref1(:,1:TW_p(tw_length)));
            [wx2,wy2,r2]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),ref2(:,1:TW_p(tw_length)));
            [wx3,wy3,r3]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),ref3(:,1:TW_p(tw_length)));
            [wx4,wy4,r4]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),ref4(:,1:TW_p(tw_length)));
            [wx5,wy5,r5]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),ref5(:,1:TW_p(tw_length)));
            [wx6,wy6,r6]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),ref6(:,1:TW_p(tw_length)));
            [wx7,wy7,r7]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),ref7(:,1:TW_p(tw_length)));
            [wx8,wy8,r8]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),ref8(:,1:TW_p(tw_length)));
            [v,idx]=max([max(r1),max(r2),max(r3),max(r4),max(r5),max(r6),max(r7),max(r8)]);
            if idx==j
                n_correct(1,tw_length)=n_correct(1,tw_length)+1;
            end
        end
    end

end


%% MsetCCA for SSVEP recognition
K=1;    % number of extracted components for each spatial filter
for run=1:24
    idx_traindata=1:n_run;
    idx_traindata(run)=[];
    for tw_length=1:4       % time window length:  1s:1s:4s
        fprintf('MsetCCA Processi ng... TW %fs, No.crossvalidation %d \n',TW(tw_length),run);
        % Reference signals optimization by MsetCCA
        %mohasebe W ha bRaye har freq
        Temp1=zeros(19*K,TW_p(tw_length)); Temp2=Temp1; Temp3=Temp2;  Temp4=Temp3; Temp5=Temp4;Temp6=Temp5;Temp7=Temp6;Temp8=Temp7;
        W1=msetcca(SSVEPdata3(:,1:TW_p(tw_length),idx_traindata,1),K);
        W2=msetcca(SSVEPdata3(:,1:TW_p(tw_length),idx_traindata,2),K);
        W3=msetcca(SSVEPdata3(:,1:TW_p(tw_length),idx_traindata,3),K);
        W4=msetcca(SSVEPdata3(:,1:TW_p(tw_length),idx_traindata,4),K);
        W5=msetcca(SSVEPdata3(:,1:TW_p(tw_length),idx_traindata,5),K);
        W6=msetcca(SSVEPdata3(:,1:TW_p(tw_length),idx_traindata,6),K);
        W7=msetcca(SSVEPdata3(:,1:TW_p(tw_length),idx_traindata,7),K);
        W8=msetcca(SSVEPdata3(:,1:TW_p(tw_length),idx_traindata,8),K);
        for qq=1:23 %sakhtane ref ha ba estefade az W ha
            Temp1((qq-1)*K+1:qq*K,:)=W1(:,:,qq)'*SSVEPdata3(:,1:TW_p(tw_length),idx_traindata(qq),1);
            Temp2((qq-1)*K+1:qq*K,:)=W2(:,:,qq)'*SSVEPdata3(:,1:TW_p(tw_length),idx_traindata(qq),2);
            Temp3((qq-1)*K+1:qq*K,:)=W3(:,:,qq)'*SSVEPdata3(:,1:TW_p(tw_length),idx_traindata(qq),3);
            Temp4((qq-1)*K+1:qq*K,:)=W4(:,:,qq)'*SSVEPdata3(:,1:TW_p(tw_length),idx_traindata(qq),4);
            Temp5((qq-1)*K+1:qq*K,:)=W5(:,:,qq)'*SSVEPdata3(:,1:TW_p(tw_length),idx_traindata(qq),5);
            Temp6((qq-1)*K+1:qq*K,:)=W6(:,:,qq)'*SSVEPdata3(:,1:TW_p(tw_length),idx_traindata(qq),6);
            Temp7((qq-1)*K+1:qq*K,:)=W7(:,:,qq)'*SSVEPdata3(:,1:TW_p(tw_length),idx_traindata(qq),7);
            Temp8((qq-1)*K+1:qq*K,:)=W8(:,:,qq)'*SSVEPdata3(:,1:TW_p(tw_length),idx_traindata(qq),8);
        end
        
        % Recognition mohasebe CCA bein test(trial kenar gozashte) va ref
        % haye jadid
        for j=1:8
            [wx1,wy1,r1]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),Temp1);
            [wx2,wy2,r2]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),Temp2);
            [wx3,wy3,r3]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),Temp3);
            [wx4,wy4,r4]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),Temp4);
            [wx5,wy5,r5]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),Temp5);
            [wx6,wy6,r6]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),Temp6);
            [wx7,wy7,r7]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),Temp7);
            [wx8,wy8,r8]=cca(SSVEPdata3(:,1:TW_p(tw_length),run,j),Temp8);
            [v,idx]=max([max(r1),max(r2),max(r3),max(r4),max(r5),max(r6),max(r7),max(r8)]);
            if idx==j
                n_correct(2,tw_length)=n_correct(2,tw_length)+1;
            end
            if tw_length==4
                if idx==j
                    n_correct_freq(j)=n_correct_freq(j)+1;
                end
            end
        end
    end
    
end


%% Plot accuracy
accuracy=100*n_correct/n_sti/24;
accuracy_per_freq=100*n_correct_freq/24;
col={'b-*','r-o'};
for mth=2:2
    plot(TW,accuracy(mth,:),col{mth},'LineWidth',1);
    hold on;
end
xlabel('Time window length (s)');
ylabel('Accuracy (%)');
grid;
xlim([0.75 5.25]);
ylim([0 100]);
set(gca,'xtick',1:5,'xticklabel',1:5);
title('\bf MsetCCA');
h=legend({'MsetCCA'});
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

%% mset 
function W=msetcca(X,K)
 
% vorudi:  X -- EEG data (channel x point x trial)
%         K -- Number of extracted joint spatial filters
% khoruji: W -- Joint spatial filters in the columns


nchannel=size(X,1);
W=zeros(size(X,1),K,size(X,3));
N_trial=size(X,3);

% normlize kardan
V=zeros(nchannel,nchannel);
for n=1:N_trial
    Xwhit=X(:,:,n);
    npot=size(Xwhit,2);
    Xwhit=Xwhit-repmat(mean(Xwhit,2),1,npot);
    C=Xwhit*Xwhit'/npot;
    [vec,val]=eig(C);
    V(:,:,n)=sqrt(val)\vec';
    X(:,:,n)=V(:,:,n)*Xwhit;
end

% Multiset CCA for learning joint spatial filters W
Y=[];
for n=1:N_trial
    Y=[Y;X(:,:,n)];
end
R=cov(Y.');
S=diag(diag(R));
[tempW rho]=eigs(R-S,S,K);
for n=1:N_trial %baraye hame trial ha poshte hame pas be tedad channel ha baraye har trial joda mikonim
    W(:,:,n)=tempW((n-1)*nchannel+1:n*nchannel,:)./norm(tempW((n-1)*nchannel+1:n*nchannel,:));
end
for n=1:N_trial %normlize kardan
    W(:,:,n)=(W(:,:,n)'*V(:,:,n))';
end
end


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

