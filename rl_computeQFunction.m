
% rl_computeQFunction.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               %
% COMP-652 Machine Learning     %
% Ryan Faulkner - 260310308     %    
%                               %
% MSc Thesis                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DESCRIPTION:
%
% Computes the Q values - assume that there are only two actions
%

function Q = rl_computeQFunction(transitionMatrix, policy, rewards, gamma)

% transitionMatrix - a cell containing the transition kerenels for each action

alpha = 0.01;
numSteps = 100000;
numStates = size(transitionMatrix,1);
numActions = 2;

Q = zeros(numStates, numActions);

for i = 1:numSteps
    
    % Randomly choose a state and action
    % ==================================
    s1 = ceil(numStates * rand);    
    a1 = ceil(numActions * rand);        
    
    dist = transitionMatrix{a1}(s1,:);
    sample = rand; 
    sum = dist(1); 
    count = 1;
    
    while sum < sample && count < numStates
        count = count + 1;
        sum = sum + dist(count);
    end

    s2 = count;
    a2 = policy(s2);    % choose the next action according to the policy
    
    Q(s1,a1) = Q(s1,a1) + alpha * (rewards(s2) + gamma * Q(s2,a2) - Q(s1,a1));
end

