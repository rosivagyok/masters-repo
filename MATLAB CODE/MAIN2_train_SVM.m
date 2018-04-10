clc 
clear all

load('E:\MATLAB\Project\Project\Cross\features.mat');
load('E:\MATLAB\Project\Project\Cross\train.mat');
load('E:\MATLAB\Project\Project\Cross\test.mat');
load('E:\MATLAB\Project\Project\Cross\gt_train.mat');
load('E:\MATLAB\Project\Project\Cross\gt_test.mat');

model1 = svmtrain(gt_train.gt_train1, train.train1(:,:), '-s 0 -t 3 -c 0.9 -b 1 -g 0.00001');
model2 = svmtrain(gt_train.gt_train2, train.train2(:,:), '-s 0 -t 3 -c 0.9 -b 1 -g 0.00001');
model3 = svmtrain(gt_train.gt_train3, train.train3(:,:), '-s 0 -t 3 -c 0.9 -b 1 -g 0.00001');
model4 = svmtrain(gt_train.gt_train4, train.train4(:,:), '-s 0 -t 3 -c 0.9 -b 1 -g 0.00001');

[predict_label1, accuracy1, prob_values1] = svmpredict(gt_test.gt_test1, test.test1, model1, '-b 1');
[predict_label2, accuracy2, prob_values2] = svmpredict(gt_test.gt_test2, test.test2, model2, '-b 1');
[predict_label3, accuracy3, prob_values3] = svmpredict(gt_test.gt_test3(1:2879), test.test3, model3, '-b 1');
[predict_label4, accuracy4, prob_values4] = svmpredict(gt_test.gt_test4, test.test4, model4, '-b 1');

total_accuracy = [accuracy1; accuracy2; accuracy3; accuracy4];

accuracy_final = max(total_accuracy)