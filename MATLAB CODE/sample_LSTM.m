clc
clear all

labels = table2array(readtable('C:\Users\pro\Downloads\labels.xlsx'));

seq_size = 8;

idx0 = find(labels==0);
idx1 = find(labels==1);
idx2 = find(labels==2);

count = min([length(idx0),length(idx1),length(idx2)]);

idx0 = idx0(1:count);
idx1 = idx1(1:count);
idx2 = idx2(1:count);

idx0_labels = labels(idx0);
idx1_labels = labels(idx1);
idx2_labels = labels(idx2);

idx0_labels = idx0_labels(1:count);
idx1_labels = idx1_labels(1:count);
idx2_labels = idx2_labels(1:count);

%Create 0 sequences
range0 = [];
myval1 = 0;
myval2 = 0;
for i = 1:length(idx0)
    if i == length(idx0) - seq_size - 1
        break;
    end
    myval1 = idx0(i);
    myval2 = idx0(i+seq_size);
    if ~ismember(myval1, range0)
        if myval2 - myval1 == seq_size
            range0 = [range0; idx0(i:i+seq_size)];
        end
    end
   
end
range0 = reshape(range0(1:length(range0)-rem(length(range0),seq_size)),[seq_size,(length(range0)-rem(length(range0),seq_size))/seq_size]);

%Create 1 sequences
range1 = [];
myval1 = 0;
myval2 = 0;
for i = 1:length(idx1)
    if i == length(idx1) - seq_size - 1
        break;
    end
    myval1 = idx1(i);
    myval2 = idx1(i+seq_size);
    if ~ismember(myval1, range1)
        if myval2 - myval1 == seq_size
            range1 = [range1; idx1(i:i+seq_size)];
        end
    end
    
end
range1 = reshape(range1(1:length(range1)-rem(length(range1),seq_size)),[seq_size,(length(range1)-rem(length(range1),seq_size))/seq_size]);

%Create 2 sequences
range2 = [];
myval1 = 0;
myval2 = 0;
for i = 1:length(idx2)
    if i == length(idx2) - seq_size - 1
        break;
    end
    myval1 = idx2(i);
    myval2 = idx2(i+seq_size);
    if ~ismember(myval1, range2)
        if myval2 - myval1 == seq_size
            range2 = [range2; idx2(i:i+seq_size)];
        end
    end
    
end
range2 = reshape(range2(1:length(range2)-rem(length(range2),seq_size)),[seq_size,(length(range2)-rem(length(range2),seq_size))/seq_size]);