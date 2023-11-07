%=========================================================================
clc;clear;close all
% addpath('.\libsvm-3.21\matlab');
load('.\IndianPines\useful_sp_lab')%%Indian Pines
%load('.\KSC\useful_sp_lab')%%KSC
%load('.\paviaU\useful_sp_lab')%%PaviaU
addpath(genpath(cd));

num_PC           =   200;  
%num_PC           =   103;  
%num_PC           =   176;
%num_Pixel        =   1; % THE OPTIMAL Number of Superpixel. Indian:100, PaviaU:20, KSC:1
num_Pixel        =   100;

database         =   'Indian';
%database         =   'PaviaU';
%database         =   'KSC';

%% load the HSI dataset
if strcmp(database,'Indian')
    load ..\Indian_pines_corrected;load ..\Indian_pines_gt;
    data3D = indian_pines_corrected;        label_gt = indian_pines_gt;
    num_Pixel        =   100;
elseif strcmp(database,'PaviaU')    
    load ..\PaviaU.mat;load ..\PaviaU_gt.mat;
    data3D = paviaU;        label_gt = paviaU_gt;
    num_Pixel        =  20;
elseif strcmp(database,'KSC')    
    load ..\KSC.mat;load ..\KSC_gt.mat;
    data3D = KSC;        label_gt = KSC_gt;
    num_Pixel        =   1;
end
data3D = data3D./max(data3D(:));
%% super-pixels segmentation
labels = cubseg(data3D,num_Pixel);
%[PaviaU_gyh] = SuperPCA(data3D,num_PC,labels);
%[KSC_gyh] = SuperPCA(data3D,num_PC,labels);
[IP_gyh] = SuperPCA(data3D,num_PC,labels);
%% Calculate dynamic weight
% B1 = KSC_gyh(:,:,1);
% [M,N,B] = size(KSC_gyh);
% IP_gyh_2d = reshape(KSC_gyh,M*N,B);
% B1 = IP_gyh(:,:,1);
% [M,N,B] = size(IP_gyh);
% IP_gyh_2d = reshape(IP_gyh,M*N,B);
B1 = IP_gyh(:,:,1);
[M,N,B] = size(IP_gyh);
IP_gyh_2d = reshape(IP_gyh,M*N,B);
useful_sp_lab_1d = reshape(useful_sp_lab,M*N,1);
for i=1:size(IP_gyh_2d,2)
    IP_add = [IP_gyh_2d,useful_sp_lab_1d];
    for j = 0:max(useful_sp_lab_1d)
        samelist=IP_add(IP_add(:,B+1)==j,:);
        [m,n] = size(samelist);
        R_sum = sum(samelist);
        M1 = R_sum/m;
        M_(j+1,:) = M1(1,:);
    end
end
for k = 1:size(IP_gyh_2d,2)
    per = IP_add(find(IP_add(:,B+1)==0),:);
    D_ = repmat(M_(1,:),size(per,1),1);
    temp1 = size(per,1);
    temp2 = size(D_,1);
    P(1:temp1,:) = per;
    for l=1:size(M_,1)-1
        permutation = IP_add(find(IP_add(:,B+1)==l),:);
        P(temp1+1:temp1+size(permutation,1),:) = permutation;
        temp1 = temp1+size(permutation,1);
        K_ = repmat(M_(l+1,:),size(permutation,1),1);
        D(temp2+1:temp2+size(K_,1),:) = K_;
        temp2 = temp2+size(K_,1);
    end
end
D(1:size(D_),:) = D_;
D_ = P-D;
D__ = D_.^2;
D__(:,B+1) = D(:,B+1);
for r = 1:size(D__,2)
    for r1 = 0:size(M_,1)-1
        perm =D__(find(D__(:,B+1)==r1),:);
        perm_sum = sum(perm);
        p = sqrt(perm_sum/size(perm,1));
        p_(r1+1,:) = p(1,:);
    end
end
p_(:,B+1) = [];
for u = 1:size(p_,1)
    for u1 = 1:size(p_,2)
        c = sum(p_,2)/B;
        final = exp(-c);
    end
end
final = diag(final);
final(1,:) = [];
final(:,1) = [];
save('..\final.mat','final')%Innovation point 1
save('..\IP_gyh.mat','IP_gyh')
%save('..\KSC_gyh.mat','KSC_gyh')
%save('..\PaviaU_gyh.mat','PaviaU_gyh')

%load('..\PaviaU_gt.mat');
%load('..\KSC_gt.mat');
load('..\IP_gt.mat');
%% Divide training samples and test samples
%num_tr = [46,15,15,15,10,14,6,26,31,24,25,30,55]%6%KSC
num_tr = [5,143,83,24,48,73,3,48,2,97,245,59,20,126,39,9];%10IP
%num_tr = [1989,5594,630,919,403,1509,399,1105,284];%30UP
[ ~, ~, ~, ~, trpos,tepos ] = TrainTestPixel(IP_gyh, IP_gt, num_tr, 15);
%[ ~, ~, ~, ~, trpos,tepos ] = TrainTestPixel1(KSC_gyh, KSC_gt, num_tr, 15);
%[ ~, ~, ~, ~, trpos,tepos] = TrainTestPixel(PaviaU_gyh, PaviaU_gt, num_tr, 15);
save('..\trpos', 'trpos');
save('..\tepos', 'tepos');
system('python trainMDGCN.py');
