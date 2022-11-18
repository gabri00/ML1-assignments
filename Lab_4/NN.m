[wine_input, wine_target] = loadDataset('wine.data', 1, true);

% Set input and target of the NN
x = wine_input;
t = wine_target_new;
% 
% trainFcn = 'trainscg';
% 
% hiddenLayerSize = 10;
% net = patternnet(hiddenLayerSize, trainFcn);
% 
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
% 
% [net,tr] = train(net,x,t);
% 
% y = net(x);
% e = gsubtract(t,y);
% performance = perform(net,t,y)
% tind = vec2ind(t);
% yind = vec2ind(y);
% percentErrors = sum(tind ~= yind)/numel(tind);
% 
% view(net)
% 
% figure, plotconfusion(t,y)