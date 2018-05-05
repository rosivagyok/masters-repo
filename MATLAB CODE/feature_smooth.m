function [ pose_feats ] = feature_smooth( pose_feats )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

%test
% load('E:\MATLAB\Project\Project\test.mat');


%pose_feats = cell2mat(pose_feats);

for j= [1:2:11 27:2:size(pose_feats,2)]
    
    %search 0 idx
    idx = find(pose_feats(:,j)==0);

    %counters
    k = 1;
    myval1 = 0;
    myval2 = 0;
    range = [1 1];

    for i=1:length(idx)
        if i == length(idx)
            myval2 = idx(i)
            range(k,1:2) = [myval1 myval2];
        elseif idx(i + 1) - idx(i) == 1 && myval1 == 0
            myval1 = idx(i);
        elseif idx(i + 1) - idx(i) == 1
            continue;
        else
            myval2 = idx(i)
            range(k,1:2) = [myval1 myval2];
            k = k + 1;
            myval1 = 0;
            myval2 = 0;
        end

        if k > 1 && range(k-1,1) == 0
            range(k-1,1) = range(k-1,2);
        end
    end

    stupid_shit = find(range(:,1)==0);
    range(stupid_shit,1) = range(stupid_shit,2);

    for i=1:size(range,1)
        %calculate interpolation range
        pathXY = [pose_feats(range(i,1)-1,j:j+1); pose_feats(range(i,2)+1,j:j+1)]
        stepLengths = sqrt(sum(diff(pathXY,[],1).^2,2))
        stepLengths = [0; stepLengths] % add the starting point
        cumulativeLen = cumsum(stepLengths)
        finalStepLocs = linspace(0,cumulativeLen(end), 100)
        %set up interpolated coordinates for 100 precision
        finalPathXY = interp1(cumulativeLen, pathXY, finalStepLocs)

        %sample data from range
        a = [range(i,1) range(i,1)+1:range(i,2)]
%         b = floor((length(finalPathXY)/length(a))-1);
        c = finalPathXY(int64(floor(linspace(1,length(finalPathXY)-1,length(a)))),1:2);

        pose_feats(a,j:j+1) = c;
    end
end

pose_feats = mat2cell(pose_feats,1*ones(1,size(pose_feats,1)), 54);

end

