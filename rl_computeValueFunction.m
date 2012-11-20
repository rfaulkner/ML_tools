
% rl_computeValueFunction.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               %
% COMP-652 Machine Learning     %
% Ryan Faulkner - 260310308     %    
%                               %
% MSc Thesis                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DESCRIPTION:
%
% Computes the exact value function
%

function V = rl_computeValueFunction(transitionMatrix, rewards, gamma)

numStates = size(transitionMatrix,1);

% V = (I - aT)^-1 * R
V = inv(eye(numStates,numStates) - gamma * transitionMatrix) * transitionMatrix * rewards;
