wine_dataset = importdata('wine.data');
wine_input = wine_dataset(:, 2:size(wine_dataset, 2));
wine_target = wine_dataset(:, 1);

classes = unique(wine_target);
wine_target_new = zeros(size(wine_target, 1), length(classes));
for i = 1:size(wine_target_new)
    for j = 1:size(classes)
        if wine_target(i, :) == classes(j)
            wine_target_new(i, j) = 1;
        end
    end
end

clear wine_target;
clear i;
clear j;
clear wine_dataset;
% Transpose predictor and response
wine_input = wine_input';
wine_target_new = wine_target_new';

% Set input and target of the NN
x = wine_input;
t = wine_target_new;

trainFcn = 'trainscg';

hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

[net,tr] = train(net,x,t);

y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

view(net)

figure, plotconfusion(t,y)