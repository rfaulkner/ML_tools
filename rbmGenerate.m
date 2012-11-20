

% rbmGenerate.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               %
% Ryan Faulkner - 260310308     %    
%                               %
% MSc Thesis                    %
%                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DESCRIPTION:
%
% Given a model and a data set over the visibles generates the hidden
% activities (draws from P(h|v) ... v from data)
%

function [hidActs probs] = rbmGenerate(model, data, stochastic)

if nargin < 3
    stochastic = true;
end
    
numSamples = size(data,1);

% EXTRACT MODEL DATA
% ------------------

% Retrieve connections
modelWeights = model{1};
visToHid = modelWeights{1};
numHidUnits = size(visToHid,2);

% Retrieve hidden biases
modelBiases = model{2};
bias = modelBiases{2};


% GENERATE ACTIVITIES ON THE HIDDEN UNITS
% ---------------------------------------
probs = 1./(1 + exp(- data * visToHid - repmat(bias,numSamples,1)));
if stochastic
    hidActs = real(probs > rand(numSamples,numHidUnits));
else
    hidActs = probs;
end

