
% rbmGaussGenerate.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               %
% Ryan Faulkner - 260310308     %
%                               %
% MSc Thesis                    %
%                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DESCRIPTION:
%
% Generates over the visible units of a Gaussian RBM
%


function samples = rbmGaussGenerate(model, input, stochastic)

% meanGen indicates if we wish to generate using the means of the gaussian
% default behaviour is not to do this
if nargin < 3
    stochastic = true;
end

numSamples = size(input,1);
numVis = size(model{1}{1},2);  % Get the number of visibles

% EXTRACT MODEL DATA
% ------------------

% Retrieve connections
modelWeights = model{1};
weights = modelWeights{1};

% Retrieve hidden biases
modelBiases = model{2};
bias = modelBiases{2};


% SAMPLE GAUSSIAN
% ---------------

if stochastic
    samples = randn(numSamples,numVis) + repmat(bias,numSamples,1) + input * weights;
else
    samples = repmat(bias,numSamples,1) + input * weights;
end

