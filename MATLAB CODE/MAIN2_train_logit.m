clc 
clear all

load('E:\MATLAB\Project\Project\Cross\features.mat');
load('E:\MATLAB\Project\Project\Cross\train.mat');
load('E:\MATLAB\Project\Project\Cross\test.mat');
load('E:\MATLAB\Project\Project\Cross\gt_train.mat');
load('E:\MATLAB\Project\Project\Cross\gt_test.mat');

[model1, llh1] = logitMn(train.train1(:,:)', gt_train.gt_train1 + 1, 0.1);
[model2, llh2] = logitMn(train.train2(:,:)', gt_train.gt_train2 + 1, 0.1);
[model3, llh3] = logitMn(train.train3(:,:)', gt_train.gt_train3 + 1, 0.1);
[model4, llh4] = logitMn(train.train4(:,:)', gt_train.gt_train4 + 1, 0.1);

[y1, P1] = logitMnPred(model1, test.test1');
[y2, P2] = logitMnPred(model2, test.test2');
[y3, P3] = logitMnPred(model3, test.test3');
[y4, P4] = logitMnPred(model4, test.test4');

accuracy= 0;
for i=1:length(gt_test.gt_test1)
    if y1(i) == gt_test.gt_test1(i) +1
        accuracy=accuracy+1;
    end
end
accuracy1 = accuracy/length(gt_test.gt_test1);

accuracy= 0;
for i=1:length(gt_test.gt_test2)
    if y2(i) == gt_test.gt_test2(i) +1
        accuracy=accuracy+1;
    end
end
accuracy2 = accuracy/length(gt_test.gt_test2);

accuracy= 0;
for i=1:2879
    if y3(i) == gt_test.gt_test3(i) +1
        accuracy=accuracy+1;
    end
end
accuracy3 = accuracy/length(gt_test.gt_test3);

accuracy= 0;
for i=1:length(gt_test.gt_test4)
    if y4(i) == gt_test.gt_test4(i) +1
        accuracy=accuracy+1;
    end
end
accuracy4 = accuracy/length(gt_test.gt_test4);

total_accuracy = [accuracy1; accuracy2; accuracy3; accuracy4];

accuracy_final = max(total_accuracy)
