% This script assumes these variables are defined:
%
%   inputs - input data.
%   targets - target data.

%x = inputs;
%t = targets;
EOM3=xlsread('Output.xlsx');
FV=xlsread('Input.xlsx');
FV=FV';
EOM3=EOM3';

% Create a Pattern Recognition Network
hiddenLayerSize = 50;
net = patternnet(hiddenLayerSize);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
%net.input.processFcns = {'removeconstantrows','mapminmax'};
%net.output.processFcns = {'removeconstantrows','mapminmax'};


% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 80/100;
net.divideParam.testRatio = 80/100;

% For help on training function 'trainscg' type: help trainscg
% For a list of all training functions type: help nntrain
net.trainFcn = 'traingda';  % Scaled conjugate gradient
net.trainParam.epochs = 600000;
net.trainParam.lr = 0.04;
net.trainParam.max_fail = 300000;
net.performFcn = 'mse';

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
%net.performFcn = ''%crossentropy';  % Cross-entropy

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit', 'plotconfusion'};


% Train the Network
[net,tr] = train(net,FV,EOM3);

% Test the Network
y = net(FV);
e = gsubtract(EOM3,y);
tind = vec2ind(EOM3);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);
performance = perform(net,EOM3,y);

% Recalculate Training, Validation and Test Performance
trainTargets = EOM3 .* tr.trainMask{1};
valTargets = EOM3  .* tr.valMask{1};
testTargets = EOM3  .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

% View the Network
save('mynet2.mat');
view(net)