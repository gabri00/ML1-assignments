function [input, target_expanded] = loadDataset(filename, t_col, T)
%LOADDATASET Summary of this function goes here
%   Detailed explanation goes here
    dataset = importdata(filename);
    input = dataset(:, 2:size(dataset, 2));
    target = dataset(:, t_col);
    
    classes = unique(target);
    target_expanded = zeros(size(target, 1), length(classes));

    for i = 1:size(target_expanded)
        for j = 1:size(classes)
            if target(i, :) == classes(j)
                target_expanded(i, j) = 1;
            end
        end
    end

    if T
        input = input';
        target_expanded = target_expanded';
    end
end