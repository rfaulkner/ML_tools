
% rl_computeReturns.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               %
% Ryan Faulkner - 260310308     %    
%                               %
% MSc Thesis                    %
%                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DESCRIPTION:
%
% Computes the expected return of a policy
%

function R = rl_computeReturns(policy, reward, gamma, transitions)

sweeps = 10;
R = 0;
avR = 0;
state = 1;

for k = 1:100
    for i = 1:sweeps

        % Choose an action
        action = policy(state);
        transition = transitions{action};

        % SAMPLE NEW STATE from transition function
        u = rand;
        dist = transition(state,:);
        for j = 1:length(dist)
            if u < sum(dist(1:j))
                state = j;
                break;
            end
        end

        R = reward(state) + gamma * R;

    end
    avR = avR + R;
end

R = avR / 100;
