clc 
clear all

load('E:\MATLAB\Project\Project\Cross\full\features.mat');
load('E:\MATLAB\Project\Project\Cross\full\train.mat');
load('E:\MATLAB\Project\Project\Cross\full\test.mat');
load('E:\MATLAB\Project\Project\Cross\full\gt_train.mat');
load('E:\MATLAB\Project\Project\Cross\full\gt_test.mat');

NET1 = mlp(66, 130, 3, 'softmax');
[NETready1, errorall1] = mlptrain(NET1, train.train1, gt_train.gt_train1_mlp, 220);
[Y1, Z1, A1] = mlpfwd(NETready1, test.test1);

NET2 = mlp(66, 130, 3, 'softmax');
[NETready2, errorall2] = mlptrain(NET2, train.train2, gt_train.gt_train2_mlp, 220);
[Y2, Z2, A2] = mlpfwd(NETready2, test.test2);

NET3 = mlp(66, 130, 3, 'softmax');
[NETready3, errorall3] = mlptrain(NET3, train.train3, gt_train.gt_train3_mlp, 220);
[Y3, Z3, A3] = mlpfwd(NETready3, test.test3);

NET4 = mlp(66, 130, 3, 'softmax');
[NETready4, errorall4] = mlptrain(NET4, train.train4, gt_train.gt_train4_mlp, 220);
[Y4, Z4, A4] = mlpfwd(NETready4, test.test4);

[P, C] = max(Y1,[],2); 
M = C-1==gt_test.gt_test1;
acc1 = sum(M)/length(gt_test.gt_test1)

[P, C] = max(Y2,[],2); 
M = C-1==gt_test.gt_test2;
acc2 = sum(M)/length(gt_test.gt_test2)

[P, C] = max(Y3,[],2); 
M = C-1==gt_test.gt_test3(1:2879);
acc3 = sum(M)/length(gt_test.gt_test3)

[P, C] = max(Y4,[],2); 
M = C-1==gt_test.gt_test4;
acc4 = sum(M)/length(gt_test.gt_test4)

total_accuracy = [acc1; acc2; acc3; acc4];
acc_max = max(total_accuracy)
acc_average = mean(total_accuracy)