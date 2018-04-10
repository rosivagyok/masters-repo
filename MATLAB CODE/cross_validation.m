function [ test, train, gt_test, gt_train ] = cross_validation( pose_feats, labels, mode )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

%test
% load('E:\MATLAB\Project\Project\test.mat');

%normalize features
pose_feats = norm_feats(pose_feats);

%create subsets for training
train.train1 = pose_feats(1:floor(length(pose_feats)/4)*3,:);
gt_train.gt_train1 = double(labels(1:floor(length(pose_feats)/4)*3));

train.train2 = pose_feats(floor(length(pose_feats)/4):end,:);
gt_train.gt_train2 = double(labels(floor(length(pose_feats)/4):end));

train.train3 = [pose_feats(1:floor(length(pose_feats)/4),:); pose_feats(floor(length(pose_feats)/4)*2:end,:)];
gt_train.gt_train3 = [labels(1:floor(length(pose_feats)/4)); labels(floor(length(pose_feats)/4)*2:end)];

train.train4 = [pose_feats(1:floor(length(pose_feats)/4)*2,:); pose_feats(floor(length(pose_feats)/4)*3:end,:)];
gt_train.gt_train4 = [labels(1:floor(length(pose_feats)/4)*2); labels(floor(length(pose_feats)/4)*3:end)];

%create subsets for testing
test.test1 = pose_feats(floor(length(pose_feats)/4)*3:end,:);
gt_test.gt_test1 = double(labels(floor(length(pose_feats)/4)*3:end));

test.test2 = pose_feats(1:floor(length(pose_feats)/4),:);
gt_test.gt_test2 = double(labels(1:floor(length(pose_feats)/4)));

test.test3 = pose_feats(floor(length(pose_feats)/4):floor(length(pose_feats)/4)*2,:);
gt_test.gt_test3 = double(labels(floor(length(pose_feats)/4):5758));

test.test4 = pose_feats(floor(length(pose_feats)/4)*2:floor(length(pose_feats)/4)*3,:);
gt_test.gt_test4 = double(labels(floor(length(pose_feats)/4)*2:floor(length(pose_feats)/4)*3));

%only for mlp
if strcmp(mode,'mlp')
    gt_train.gt_train1_mlp = zeros(length(gt_train.gt_train1),3);
    gt_train.gt_train2_mlp = zeros(length(gt_train.gt_train2),3);
    gt_train.gt_train3_mlp = zeros(length(gt_train.gt_train3),3);
    gt_train.gt_train4_mlp = zeros(length(gt_train.gt_train4),3);
    
    gt_test.gt_test1_mlp = zeros(length(gt_test.gt_test1),3);
    gt_test.gt_test2_mlp = zeros(length(gt_test.gt_test2),3);
    gt_test.gt_test3_mlp = zeros(length(gt_test.gt_test3),3);
    gt_test.gt_test4_mlp = zeros(length(gt_test.gt_test4),3);
    
    idx0.sub1 = find(gt_train.gt_train1 == 0);
    idx0.sub2 = find(gt_train.gt_train2 == 0);
    idx0.sub3 = find(gt_train.gt_train3 == 0);
    idx0.sub4 = find(gt_train.gt_train4 == 0);
    idx1.sub1 = find(gt_train.gt_train1 == 1);
    idx1.sub2 = find(gt_train.gt_train2 == 1);
    idx1.sub3 = find(gt_train.gt_train3 == 1);
    idx1.sub4 = find(gt_train.gt_train4 == 1);
    idx2.sub1 = find(gt_train.gt_train1 == 2);
    idx2.sub2 = find(gt_train.gt_train2 == 2);
    idx2.sub3 = find(gt_train.gt_train3 == 2);
    idx2.sub4 = find(gt_train.gt_train4 == 2);
    
    
    gt_train.gt_train1_mlp(idx0.sub1,1) = 1;
    gt_train.gt_train1_mlp(idx1.sub1,2) = 1;
    gt_train.gt_train1_mlp(idx2.sub1,3) = 1;
    
    gt_train.gt_train2_mlp(idx0.sub2,1) = 1;
    gt_train.gt_train2_mlp(idx1.sub2,2) = 1;
    gt_train.gt_train2_mlp(idx2.sub2,3) = 1;
    
    gt_train.gt_train3_mlp(idx0.sub3,1) = 1;
    gt_train.gt_train3_mlp(idx1.sub3,2) = 1;
    gt_train.gt_train3_mlp(idx2.sub3,3) = 1;
    
    gt_train.gt_train4_mlp(idx0.sub4,1) = 1;
    gt_train.gt_train4_mlp(idx1.sub4,2) = 1;
    gt_train.gt_train4_mlp(idx2.sub4,3) = 1;
end

end

