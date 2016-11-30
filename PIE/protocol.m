clear; clc; close all

load Train5_64;
load fea64;
load gnd64;

fea = fea64; clear fea64;
gnd = gnd64; clear gnd64;
Train = Train5_64; clear Train5_64;

fea1 = fea;

%%% Train contains 20 random splits of the data
%%% gnd is the matrix with data labels (i.e., face identities)


error = [];
dim =40; 
%%checks recognition rate every dim dimensions (change it appropriatly for PCA, LDA etc)
%%for PCA and LDA you can use smaller steps (i.e., 100 for PCA and 10-20
%%for LDA)
for jj = 1:20
    jj

    TrainIdx = Train(jj, :);
    TestIdx = 1:size(fea, 1);
    TestIdx(TrainIdx) = [];

    fea_Train = fea1(TrainIdx,:);
    gnd_Train = gnd(TrainIdx);
    [gnd_Train ind] = sort(gnd_Train, 'ascend');
    fea_Train = fea_Train(ind, :);
    save('fea_Train', 'fea_Train')

    fea_Test = fea1(TestIdx,:);
    gnd_Test = gnd(TestIdx);

    %%%%This is where you put your function%%%%%%%%%%%%%
    %%%Currently there is no dimensionality reduction%%%
    
    [U_reduc, whiten_factor] = PCA(fea_Train', 340);
%     U_reduc = eye(size(64*64,64*64)); 

%     size(U_reduc)
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%dimensionality reduction step
    
    oldfea = (U_reduc' * center(fea_Train'))';
    newfea = (U_reduc' * center(fea_Test'))';
    
%     oldfea = fea_Train*U_reduc;
%     newfea = fea_Test*U_reduc;

    %%subtraction of the mean
    mg = mean(oldfea, 1);
    mg_oldfea = repmat(mg,  size(oldfea,1), 1);
    oldfea = oldfea - mg_oldfea;

    mg_newfea = repmat(mg,  size(newfea,1), 1);
    newfea = newfea - mg_newfea;

    %%%classification
    len     = 1:dim:size(newfea, 2);
    correct = zeros(1, length(1:dim:size(newfea, 2))); 
    for ii = 1:length(len)  %%for each dimension perform classification
        ii;
        Sample = newfea(:, 1:len(ii));
        Training = oldfea(:, 1:len(ii));
        Group = gnd_Train;
        k = 1;
        distance = 'cosine';
        Class = knnclassify(Sample, Training , Group, k, distance);

        correct(ii) = length(find(Class-gnd_Test == 0));
    end

    correct = correct./length(gnd_Test);
    error = [error; 1- correct];
  
end

plot(mean(error,1)); %%plotting the error 