function [ pose_feats ] = norm_feats( pose_feats )
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here

%test
% load('E:\MATLAB\Project\Project\test.mat');

trainsub=pose_feats(:,3:66);
trainsub(trainsub == 0) = NaN;
for i=1:64
    minc = min(trainsub(:,i));
    maxc = max(trainsub(:,i));
    trainsub(:,i) = (trainsub(:,i) - minc) / (maxc - minc);
end
trainsub(isnan(trainsub)) = -1;
pose_feats(:,3:66)=trainsub;

end

