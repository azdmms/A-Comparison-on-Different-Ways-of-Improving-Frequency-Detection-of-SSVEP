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
%% combine data haye matlub maghale dar Xelect 
%kole data matlub maghale dakhele Xelect rikhte shod
Xelect=zeros(4,8,1255,8,6);
Xelect(1,:,:,:,:)=X_elect;
Xelect(2,:,:,:,:)=X_elect1;
Xelect(3,:,:,:,:)=X_elect2;
Xelect(4,:,:,:,:)=X_elect3;
%% sakhte dade haye train koli
Xelecttrain(1,:,:,:,:)=X_elect;
Xelecttrain(2,:,:,:,:)=X_elect1;
% for i=idx
%     Xfreq3(:,:,:,:)=data3(:,:,i,:);
% end
% for j=[46,44,48,50,52,51,62,63]
%     X_elect3(:,:,:,:)=Xfreq3(j,:,:,:);
% end

% ta inja taze data shabih maghale shod

%% normal CCA
Xtest=X_elect(:,:,1,1);
freq=[8,9.2,11,12,13,13.4,15,15.8];
ref=zeros(1,1255) ;
for ij=[1:1:8]
for t=1:1255
     refnormalcca(1,t,ij)=sin(2*pi*freq(ij)*t/1000);
     refnormalcca(2,t,ij)=cos(2*pi*freq(ij)*t/1000);
end
end
ref=refnormalcca;
%% har subject yek deghat koli
for j=[1:1:6]
    correctfreq(:,j)=freq;
end
for l=[1:1:numberofsubj]
    for j=[1:1:6]
        for k=[1:1:8]
            Xtest=squeeze(Xelect(l,:,:,k,j));
            ro=zeros(1,8);
            for i=[1:1:8]
                Y = ref(:,:,i) ;
                RHO= myCCA(Xtest',Y');
                ro(1,i)=max(RHO);
            end
            [maxvalue,maxindx]=max(ro);
            determinedfreq(k,j)=freq(maxindx);
        end
        
    end
    
    determinedfreq1=determinedfreq(:);
    correctfreq1=correctfreq(:);
    s=correctfreq1==determinedfreq1;
    similarity(l) = (sum(s)/numel(s))*100;
end
subjectindx=[1:1:4];
plot(subjectindx,similarity,'-o');
    xlim([1 4]);
    ylim([0 30]);
    xlabel('subject') ;
    ylabel('accuracy') ;
    title('overall accuracy of subjects') ; 
%% deghat haye marbut be har frequency subject
for j=[1:1:6]
        correctfreq(:,j)=freq;
end
for l=[1:1:numberofsubj]
    for k=[1:1:8] 
        for j=[1:1:6]
            Xtest=squeeze(Xelect(l,:,:,k,j));
            ro=zeros(1,8);
            for i=[1:1:8]
                Y = ref(:,:,i) ;
                RHO= myCCA(Xtest',Y');
                ro(1,i)=max(RHO);
            end
            [maxvalue,maxindx]=max(ro);
            determinedfreq(k,j)=freq(maxindx);
        end
        determinedfreq1=determinedfreq(k,:);
        correctfreq1=correctfreq(k,:);
        s=correctfreq1==determinedfreq1;
        similarity2(l,k) = (sum(s)/numel(s))*100;
    end
    
    
end
for i=1:1:4
    subplot(2,2,i);
    plot(freq,similarity2(i,:),'-o');
    xlim([6 18]);
    ylim([0 60]);
    xlabel('frequency') ;
    ylabel('accuracy') ;
    title(['subject' , num2str(i)]) ; 
end
%% sakhte reference aval

%dar in bakhsh do nafar aval dade train va do nafar dovom dade test hastand

ref1=zeros(8,1255,8) ;
for k=1:1:8
    for l=1:1:6
        ref1(:,:,k)=ref1(:,:,k)+X_elect(:,:,k,l)+X_elect1(:,:,k,l);%+X_elect2(:,:,k,l)+X_elect3(1,:,k,l);
    end
end
ref1=ref1/12;
%% sakhte reference seri dovom
j=1;
for h=1:1:8
    for b=1:1:6
        
        RHO2= myCCA(squeeze(Xelect(1,:,:,h,b)),ref1(:,:,h));
        ro2(h,b)=max(RHO2);
        
        
    end
    
end
sortro2 = sort(ro2,2);
for h=1:1:8
for m=1:1:6
ans1 = find(sortro2(h,m)==ro2(h,:));
ref2(:,:,h,m)=Xelect(1,:,:,h,ans1);
end
end
% ref2(:,:,h)=X(:,:,h,b);%rikhtane har signal dakhele X yani har trial har block
%         j=j+1;
% i=1;
%% estefade az reference ha
% entekhabe 3 reference az seri dovom va seri aval
for h=1:1:8
ref2final(1,:,:,h)= ref2(:,:,h,1);
ref2final(2,:,:,h)= ref2(:,:,h,2);
ref2final(3,:,:,h)= ref2(:,:,h,3);
end
ref1sum=zeros(1255,8);
for k=1:1:8
    
        ref1sum(:,:)=ref1sum(:,:)+squeeze(ref1(k,:,:));
    
end
ref1sum=ref1sum/8;
ref2sum=zeros(3,1255,8);
for n=1:1:3
for k=1:1:8
    
        ref2sum(n,:,:)=squeeze(ref2sum(n,:,:))+squeeze(ref2final(n,k,:,:));
end  
ref2sum(n,:,:)=ref2sum(n,:,:)/8;
end
reftotal(1,:,:)=ref2sum(1,:,:);
reftotal(2,:,:)=ref2sum(2,:,:);
reftotal(3,:,:)=ref2sum(3,:,:);
reftotal(4,:,:)=ref1sum(:,:);

for j=[1:1:6]
        correctfreq(:,j)=freq;
end
for l=[3:1:numberofsubj]
    for k=[1:1:8] 
        for j=[1:1:6]
            Xtest=squeeze(Xelect(l,:,:,k,j));
            ro=zeros(1,8);
            for i=[1:1:8]
                Y = reftotal(:,:,i) ;
                RHO= myCCA(Xtest',Y');
                ro(1,i)=max(RHO);
            end
            [maxvalue,maxindx]=max(ro);
            determinedfreq(k,j)=freq(maxindx);
        end
        determinedfreq1=determinedfreq(k,:);
        correctfreq1=correctfreq(k,:);
        s=correctfreq1==determinedfreq1;
        similarity2(l,k) = (sum(s)/numel(s))*100;
    end
  
end

for i=1:1:2
    subplot(1,2,i);
    plot(freq,similarity2(i,:),'-o');
    xlim([6 18]);
    ylim([0 100]);
    xlabel('frequency') ;
    ylabel('accuracy') ;
    title(['subject' , num2str(i)]) ; 
end




%%
for j=[1:1:6]
    correctfreq(:,j)=freq;
end
for l=[1:1:numberofsubj]
    for j=[1:1:6]
        for k=[1:1:8]
            Xtest=squeeze(Xelect(l,:,:,k,j));
            ro=zeros(1,8);
            for i=[1:1:8]
                Y = reftotal(:,:,i) ;
                RHO= myCCA(Xtest',Y');
                ro(1,i)=max(RHO);
            end
            [maxvalue,maxindx]=max(ro);
            determinedfreq(k,j)=freq(maxindx);
        end
        
    end
    
    determinedfreq1=determinedfreq(:);
    correctfreq1=correctfreq(:);
    s=correctfreq1==determinedfreq1;
    similarity(l) = (sum(s)/numel(s))*100;
end
subjectindx=[1:1:4];
plot(subjectindx,similarity,'-o');
    xlim([1 4]);
    ylim([0 100]);
    xlabel('subject') ;
    ylabel('accuracy') ;
    title('overall accuracy of subjects') ; 
%% 
t_length=5;                              % data length (4 s)
TW=1:1:t_length;
simtw=[13,13,18,20,20];
plot(TW,simtw,col{mth},'LineWidth',1);
xlabel('Time window length (s)');
ylabel('Accuracy (%)');
grid;
xlim([0.75 5.25]);
ylim([0 100]);
title('normal CCA for each time window') ; 
set(gca,'xtick',1:5,'xticklabel',1:5); 

simtwadaptive=[20,30,38,45,46];
plot(TW,simtwadaptive,col{mth},'LineWidth',1);
xlabel('Time window length (s)');
ylabel('Accuracy (%)');
grid;
xlim([0.75 5.25]);
ylim([0 100]);
title('data adaptive for each time window') ; 
set(gca,'xtick',1:5,'xticklabel',1:5); 
%ro calculation
% ro=(Wx'*X*Y'*Wy)/sqrt(Wx'*(X*X')*Wx*Wy'*(Y*Y')*Wy);

%%
% Y = ref(:,:,1) ;
% [Wx, Wy, r] = cca(X,Y)
% for x=[ght(t)];
%     g=ssvep



%% functions
%% 
function [rho] = myCCA(A,B)
%% Cannonical Correlation Analysis
if size(A,2) <=  size(B,2)
    X=A;
    Y=B;
else
    X=B;
    Y=A;
end
%% step 2: calculate covariance matrixs
XY= [X,Y];
Cv= cov(XY);
p= size(X,2);
Cxx= Cv(1:p,1:p);
Cyy= Cv(p+1:end,p+1:end);
Cxy= Cv(1:p,p+1:end);
% Cyx= Cxy';
Cyx= Cv(p+1:end,1:p);
%% step 3: build your eigen value decomposition problem
C= inv(Cyy+eps) * Cyx * inv(Cxx+eps) * Cxy;
%% step 4:  eigen value decomposition
[~,D]= eig(C);
%% step 5: diag,sort, sqrt
D= diag(real(D));
D= sort(D,'descend');
rho= sqrt(D(1:p));
end

%%
function [Wx, Wy, r] = cca(X,Y)

% CCA calculate canonical correlations
%
% [Wx Wy r] = cca(X,Y) where Wx and Wy contains the canonical correlation
% vectors as columns and r is a vector with corresponding canonical
% correlations.
%
% Update 31/01/05 added bug handling.

if (nargin ~= 2)
  disp('Inocorrect number of inputs');
  help cca;
  Wx = 0; Wy = 0; r = 0;
  return;
end


% calculating the covariance matrices
z = [X; Y];
C = cov(z.');
sx = size(X,1);
sy = size(Y,1);
Cxx = C(1:sx, 1:sx) + 10^(-8)*eye(sx);
Cxx = C(1:sx, 1:sx) + 10^(-8)*eye(sx);
Cxy = C(1:sx, sx+1:sx+sy);
Cyx = Cxy';
Cyy = C(sx+1:sx+sy,sx+1:sx+sy) + 10^(-8)*eye(sy);

%calculating the Wx cca matrix
Rx = chol(Cxx);
invRx = inv(Rx);
Z = invRx'*Cxy*(Cyy\Cyx)*invRx;
Z = 0.5*(Z' + Z);  % making sure that Z is a symmetric matrix
[Wx,r] = eig(Z);   % basis in h (X)
r = sqrt(real(r)); % as the original r we get is lamda^2
Wx = invRx * Wx;   % actual Wx values

% calculating Wy
Wy = (Cyy\Cyx) * Wx; 

% by dividing it by lamda
Wy = Wy./repmat(diag(r)',sy,1);
end