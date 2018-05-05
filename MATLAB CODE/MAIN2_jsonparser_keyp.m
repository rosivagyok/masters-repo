clc
clear all

myfolder = 'E:\keypoints\full';
b = dir([myfolder '/*.json']);
for i=1:size(dir([myfolder '/*.json']),1)
    listfile = loadjson(['E:\keypoints\full\' b(i).name]);
    if ~isempty(listfile.people)
        for j=1:size(listfile.people,2)
            
            if nnz(listfile.people{1,j}.pose_keypoints) ~= 0
                a_pose{i,j} = listfile.people{1,j}.pose_keypoints;
            else
                a_pose{i,1} = zeros(1,54);
            end
            
            if nnz(listfile.people{1,j}.face_keypoints) ~= 0
                a_face{i,j} = listfile.people{1,j}.face_keypoints;
            else
                a_face{i,j} = zeros(1,210);
            end
        end
    else
        a_pose{i,1} = zeros(1,54);
        a_face{i,j} = zeros(1,210);
    end
    
    % Similarity check for false positive detections;
    % check which candidate yields more keypoints, use the one that has
    % more
    if size(listfile.people,2) > 1
        k = nnz([a_pose{i,1}(1:2) a_pose{i,1}(4:5) a_pose{i,1}(43:44) a_pose{i,1}(46:47) a_pose{i,1}(7:8) a_pose{i,1}(16:17)]);
        s = nnz([a_pose{i,2}(1:2) a_pose{i,2}(4:5) a_pose{i,2}(43:44) a_pose{i,2}(46:47) a_pose{i,2}(7:8) a_pose{i,2}(16:17)]);
        if k >= s
            a_pose{i,2} = [];
        else
            a_pose(i,1) = a_pose(i,2);
            a_pose{i,2} = [];
        end
    end
end

for i = 1:size(a_pose,1)
    % (1-2 Nose, 3-4 Neck)
    pose_feats{i,1}(1:4) = [a_pose{i,1}(1:2) a_pose{i,1}(4:5)]; 
    % (43-45 REye  45-48 LEye)
    pose_feats{i,1}(5:8) = [a_pose{i,1}(43:44) a_pose{i,1}(46:47)]; 
    % RS
    pose_feats{i,1}(9:10) = a_pose{i,1}(7:8); 
    % LS
    pose_feats{i,1}(11:12) = a_pose{i,1}(16:17);
    % REye_refined
    pose_feats{i,1}(27:40) = [a_face{i,1}(205:206) a_face{i,1}(109:110) a_face{i,1}(112:113)...
        a_face{i,1}(115:116) a_face{i,1}(118:119) a_face{i,1}(121:122) a_face{i,1}(124:125)]; 
    % LEye_refined
    pose_feats{i,1}(41:54) = [a_face{i,1}(208:209) a_face{i,1}(127:128) a_face{i,1}(130:131)...
        a_face{i,1}(133:134) a_face{i,1}(136:137) a_face{i,1}(139:140) a_face{i,1}(142:143)]; 
    
    %facial keypoints if nose, REye or LEye is missing
    if any(pose_feats{i,1}(1:2)) == 0; pose_feats{i,1}(1:2) = a_face{i,1}(91:92);end
    if any(pose_feats{i,1}(5:6)) == 0; pose_feats{i,1}(5:6) = a_face{i,1}(205:206); end  
    if any(pose_feats{i,1}(7:8)) == 0; pose_feats{i,1}(7:8) = a_face{i,1}(208:209); end
    
end

pose_feats = feature_smooth(pose_feats);


for i=1:size(a_pose,1)
    
    % Recalculate coordinates to nose origin
    if any(pose_feats{i,1}(3:4)) ~= 0
        pose_feats{i,1}(3:4) = pose_feats{i,1}(3:4) - pose_feats{i,1}(1:2);
    end
   
    if any(pose_feats{i,1}(5:6)) ~= 0
        pose_feats{i,1}(5:6) = pose_feats{i,1}(5:6) - pose_feats{i,1}(1:2);
    end
   
   
    if any(pose_feats{i,1}(7:8)) ~= 0
        pose_feats{i,1}(7:8) = pose_feats{i,1}(7:8) - pose_feats{i,1}(1:2);
    end

    if any(pose_feats{i,1}(9:10)) ~= 0
        pose_feats{i,1}(9:10) = pose_feats{i,1}(9:10) - pose_feats{i,1}(1:2);
    end
   
    if any(pose_feats{i,1}(11:12)) ~= 0
        pose_feats{i,1}(11:12) = pose_feats{i,1}(11:12) - pose_feats{i,1}(1:2);
    end
    pose_feats{i,1}(1:2) = [0 0];
   
    pose_feats{i,1}(13) = abs(pdist([pose_feats{i,1}(1:2); pose_feats{i,1}(5:6)],'euclidean'));
    pose_feats{i,1}(14) = abs(pdist([pose_feats{i,1}(1:2); pose_feats{i,1}(7:8)],'euclidean'));
    pose_feats{i,1}(15) = abs(pdist([pose_feats{i,1}(5:6); pose_feats{i,1}(7:8)],'euclidean'));
    
    % Euclidean distance between neck and all face features.
    % If neck was not found, set all 0.
    if any(pose_feats{i,1}(3:4)) ~= 0
        pose_feats{i,1}(16) = abs(pdist([pose_feats{i,1}(3:4); pose_feats{i,1}(1:2)],'euclidean'));
        pose_feats{i,1}(17) = abs(pdist([pose_feats{i,1}(3:4); pose_feats{i,1}(5:6)],'euclidean'));
        pose_feats{i,1}(18) = abs(pdist([pose_feats{i,1}(3:4); pose_feats{i,1}(7:8)],'euclidean'));
        
    else
        pose_feats{i,1}(16:18) = zeros(1,3);
        
    end
    
    % Euclidean distance between RShoulder and all face features.
    % If RShoulder was not found, set all 0.
    if any(pose_feats{i,1}(9:10)) ~= 0
        pose_feats{i,1}(19) = abs(pdist([pose_feats{i,1}(9:10); pose_feats{i,1}(1:2)],'euclidean'));
        pose_feats{i,1}(20) = abs(pdist([pose_feats{i,1}(9:10); pose_feats{i,1}(5:6)],'euclidean'));
        pose_feats{i,1}(21) = abs(pdist([pose_feats{i,1}(9:10); pose_feats{i,1}(7:8)],'euclidean'));
       
    else
        pose_feats{i,1}(19:21) = zeros(1,3);
        
    end
    
    % Euclidean distance between LShoulder and all face features.
    % If LShoulder was not found, set all 0.
    if any(pose_feats{i,1}(11:12)) ~= 0
        pose_feats{i,1}(22) = abs(pdist([pose_feats{i,1}(11:12); pose_feats{i,1}(1:2)],'euclidean'));
        pose_feats{i,1}(23) = abs(pdist([pose_feats{i,1}(11:12); pose_feats{i,1}(5:6)],'euclidean'));
        pose_feats{i,1}(24) = abs(pdist([pose_feats{i,1}(11:12); pose_feats{i,1}(7:8)],'euclidean'));
       
    else
        pose_feats{i,1}(22:24) = zeros(1,3);
       
    end
    
    % Angle between vec(neck,nose) and vec(neck,LShoulder)
    if any(pose_feats{i,1}(3:4)) ~= 0 && any(pose_feats{i,1}(11:12)) ~= 0
        pose_feats{i,1}(25) = atan2d(norm(cross([(pose_feats{i,1}(3) - pose_feats{i,1}(1)), ((pose_feats{i,1}(4) - pose_feats{i,1}(2))),0],[(pose_feats{i,1}(3) - pose_feats{i,1}(11)), ((pose_feats{i,1}(4) - pose_feats{i,1}(12))),0])),...
            dot([(pose_feats{i,1}(3) - pose_feats{i,1}(1)), ((pose_feats{i,1}(4) - pose_feats{i,1}(2))),0],[(pose_feats{i,1}(3) - pose_feats{i,1}(11)), ((pose_feats{i,1}(4) - pose_feats{i,1}(12))),0]));
    else
        pose_feats{i,1}(25) = 0;
    end
    
    % Angle between vec(neck,nose) and vec(neck,RShoulder)
    if any(pose_feats{i,1}(3:4)) ~= 0 && any(pose_feats{i,1}(9:10)) ~= 0
        pose_feats{i,1}(26) = atan2d(norm(cross([(pose_feats{i,1}(3) - pose_feats{i,1}(1)), ((pose_feats{i,1}(4) - pose_feats{i,1}(2))),0],[(pose_feats{i,1}(3) - pose_feats{i,1}(11)), ((pose_feats{i,1}(4) - pose_feats{i,1}(12))),0])),...
            dot([(pose_feats{i,1}(3) - pose_feats{i,1}(1)), ((pose_feats{i,1}(4) - pose_feats{i,1}(2))),0],[(pose_feats{i,1}(3) - pose_feats{i,1}(9)), ((pose_feats{i,1}(4) - pose_feats{i,1}(10))),0]));
    else
        pose_feats{i,1}(26) = 0;
    end

    pose_feats{i,1}(55) = abs(pdist([pose_feats{i,1}(27:28); pose_feats{i,1}(29:30)],'euclidean'));
    pose_feats{i,1}(56) = abs(pdist([pose_feats{i,1}(27:28); pose_feats{i,1}(31:32)],'euclidean'));
    pose_feats{i,1}(57) = abs(pdist([pose_feats{i,1}(27:28); pose_feats{i,1}(33:34)],'euclidean'));
    pose_feats{i,1}(58) = abs(pdist([pose_feats{i,1}(27:28); pose_feats{i,1}(35:36)],'euclidean'));
    pose_feats{i,1}(59) = abs(pdist([pose_feats{i,1}(27:28); pose_feats{i,1}(37:38)],'euclidean'));
    pose_feats{i,1}(60) = abs(pdist([pose_feats{i,1}(27:28); pose_feats{i,1}(39:40)],'euclidean'));
    pose_feats{i,1}(61) = abs(pdist([pose_feats{i,1}(41:42); pose_feats{i,1}(43:44)],'euclidean'));
    pose_feats{i,1}(62) = abs(pdist([pose_feats{i,1}(41:42); pose_feats{i,1}(45:46)],'euclidean'));
    pose_feats{i,1}(63) = abs(pdist([pose_feats{i,1}(41:42); pose_feats{i,1}(47:48)],'euclidean'));
    pose_feats{i,1}(64) = abs(pdist([pose_feats{i,1}(41:42); pose_feats{i,1}(49:50)],'euclidean'));
    pose_feats{i,1}(65) = abs(pdist([pose_feats{i,1}(41:42); pose_feats{i,1}(51:52)],'euclidean'));
    pose_feats{i,1}(66) = abs(pdist([pose_feats{i,1}(41:42); pose_feats{i,1}(53:54)],'euclidean'));
end
pose_feats = cell2mat(pose_feats);
load('E:\MATLAB\Project\Project\labels_pandora.mat');
[ test, train, gt_test, gt_train ] = cross_validation( pose_feats, labels, 'mlp' );

return;