%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           ATTENTION_LAB                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image-labeling tool for labeling attention level of a user             %
% Input: set of images to hand-label attention level                     %
% Output: two .txt file with frame name and att. level, respectively     %
% Three different attention levels:                                      %
%       - Low level of attention: 0                                      %
%       - Mid level of attention: 1                                      %
%       - High level of attention: 2                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Andrea Coifman & Peter Rohoska                                %
% Group 10-949, VGIS, Department of Electronic Systems                   %
% Aalborg University, Denmark                                            %
% This code is part of the long master thesis project from the authors   %
% The labeling program is available in english and spanish               %
% Used for labeling attention on the PANDORA dataset from:               %
% G. Borghi, M. Venturelli, R. Vezzani, R. Cucchiara,                    %
% POSEidon: Face-from-Depth for Driver Pose Estimation (paper)           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc
clear

path = uigetdir;
addpath path
imagefiles = dir([path  '\*.png']);
nfiles = max(size(imagefiles));

file_name = fopen('name.txt','w');
file_att = fopen('att.txt','w');
for i=1:nfiles
    name = imagefiles(i).name;
    folder = imagefiles(i).folder;
    file = fullfile(folder,name);
    imshow(file);
    fprintf(file_name,'%s \n',name); %Windows and Mac
    %cellReference = sprintf( 'A%d',i); % Only Windows
    %xlswrite('test.xlsx', cellstr(name) , 'Sheet1', cellReference); % Only Windows
    a = menu('Select the (subjective) attention level - Elija el nivel de atención (subjetivo)',...
        'Low attention level - Nivel de atención bajo',...
        'Mid attention level - Nivel de atención medio',...
        'High attention level - Nivel de atención alto');
    if a==1
        b=0;
    end
    if a==2
        b=1;
    end
    if a==3
        b=2;
    end
    %cellReference2 = sprintf( 'B%d',i); % Only Windows
    %xlswrite('test.xlsx',b,'Sheet1',cellReference2); % Only Windows
    fprintf(file_att,'%d \n',b); % Windows and Mac
    %fprintf('test.txt',b);
end
fclose(file_name);
fclose(file_att);
close all;

msg_end = msgbox('Thanks for participating! - ¡Gracias por su colaboración!');
if msg_end == 1
    return;
end