% rbmLearn.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               %
% Ryan Faulkner - 260310308     %    
%                               %
% MSc Thesis                    %
%                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DESCRIPTION:
%
% given sample data and parameters learns an RBM
%

% function [modelWeights modelBiases indices classifier] = rbmLearn(data, stateLabels, numGibbs, numHid, maxEpoch, learnRate, momentum, numBatches)
function [modelWeights modelBiases indices classifier] = rbmLearn(data, parameters, fid, stochastic)

% BY DEFAULT LEARNING IS STOCHASTIC
if nargin < 4
    stochastic = true;
end

% EXTRACT PARAMETERS
% ------------------
learnRate = parameters{1};
momentum = parameters{2};
maxEpoch = parameters{3};
numGibbs = parameters{4};
numBatches = parameters{5};
numHid = parameters{6};
stateLabels = parameters{7};
classifier = parameters{8};
model = parameters{9};
count = parameters{10};
useClass = parameters{11};

% ADAPTIVE LEARNING RATE
% evaluationInterval = parameters{10};
% increaseTerm = parameters{11};
% reductionFactor = parameters{12};


% Train the NN classifying for MNIST data if necessary
if size(classifier,1) == 0 && useClass == true
    classifier = cell(4,1);
    [classifier{1} classifier{2} classifier{3} classifier{4}] = trainClassifier(300, stateLabels, 1000, true);
end


numVis = size(data,2);
numCases = size(data,1);

indices = cell(4,1);
indices{1} = 1 : numVis;                                    % indices for bits of input (visible) vector at t
indices{2} = numVis + 1 : numVis + numHid;                  % indices for bits of input (visible) vector at t+1 

VH_Update = 0;
VBias_Update = 0;
HBias_Update = 0;


%%%%%%%%%%%%%%%%%%
% INITIALIZE MODEL
%%%%%%%%%%%%%%%%%%

modelWeights = cell(1,1);
modelBiases = cell(2,1);

alpha = .01;

if size(model,1) == 0
    modelWeights{1} = alpha * rand(numVis, numHid);

    modelBiases{1} = alpha * rand(1, numVis);
    modelBiases{2} = alpha * rand(1, numHid);
    
    model = cell(2,1);    
else
    modelWeights{1} = model{1}{1};

    modelBiases{1} = model{2}{1};
    modelBiases{2} = model{2}{2}; 
end

samples = zeros(numCases, numVis + numHid);

samples(:,indices{1}) = data(:,indices{1});
samples(:,indices{2}) = rand(numCases,numHid) > 0.5;

[batchdata batchSize] = batchify(samples, numBatches);

initTime = clock;

%%%%%%%%%%%%%% END INITIALIZE


samplesConditional = zeros(batchSize, numVis + numHid);
samplesJoint =  zeros(batchSize, numVis + numHid);

fprintf(fid,'\nRBM, visibles, %d; hiddens, %d; learning rate, %4.6f; momentum, %4.3f; gibbs Iterations, %d; batches, %d; stochastic = %d\n\n', ... 
    numVis, numHid, learnRate, momentum, numGibbs, numBatches, stochastic);

old_visToHid = modelWeights{1};
old_hidBias = modelBiases{2};
old_visBias = modelBiases{1};

         
for epoch = 1:maxEpoch

    %%%%%%%%%%%%%%%%%%%%%%%
    % Adapt Learning Rate
    %%%%%%%%%%%%%%%%%%%%%%%
    
%     if epoch == 1
%         
%         model{1} = modelWeights;  model{2} = modelBiases;
%         samples = gibbsSampleRBM(model, data, numGibbs, [], false);
%         bitwiseError = computeBitwiseError(stateLabels, classifier, data, samples{1});
%         oldError = bitwiseError;
%         
%     elseif mod(epoch,evaluationInterval) == 0
%         
%         model{1} = modelWeights;  model{2} = modelBiases;
%         samples = gibbsSampleRBM(model, data, numGibbs, [], false);
%         bitwiseError = computeBitwiseError(stateLabels, classifier, data, samples{1});
%         deltaE = oldError - bitwiseError;
%         
%         if deltaE < learnRate
%             learnRate = reductionFactor * learnRate;
%         elseif learnRate > 4 * deltaE
%             learnRate = learnRate + increaseTerm;        
%         end
%         
%         oldError = bitwiseError;    
%             
%     end


    %%%%%%%%%%%%%%%%%%%%%%%
    % Compute ERROR on selected epochs
    %%%%%%%%%%%%%%%%%%%%%%%
    
    if epoch == 2^count || epoch == maxEpoch
        
        % COMPUTE ABSOLUTE CHANGE IN WEIGHTS
        delta_visToHid = sum(sum(abs(old_visToHid - modelWeights{1}))) / (numVis * numHid);
        delta_visBias = sum(abs(old_visBias - modelBiases{1})) / (numVis);
        delta_hidBias = sum(abs(old_hidBias - modelBiases{2})) / (numHid);
        
         % set params for model 
         visToHid = modelWeights{1};
         hidBias = repmat(modelBiases{2},numCases,1);
         visBias = repmat(modelBiases{1},numCases,1);

         vis = data(:,indices{1});
         
         % compute hiddens
         probs = 1./(1 + exp(- vis * visToHid - hidBias));
         if stochastic
             hid = probs > rand(numCases, numHid);
         else
             hid = probs;
         end

         % compute visibles reconstruction
         probs = 1./(1 + exp(- hid * visToHid' - visBias));
         if stochastic
             vis = probs > rand(numCases, numVis);
         else
             vis = probs;
         end
         
         % get the current time
         currTime = clock;
         if currTime(3) > initTime(3)
            relativeTime = currTime(4)*60 + currTime(5) + (23 - initTime(4))*60 + (60 - initTime(5)); 
         else
             relativeTime = (currTime(4) - initTime(4))*60 + (currTime(5) - initTime(5)); 
         end
         
         % COMPUTE the Bitwise Error
         % =========================
         if useClass
            [bitwiseError numBad] = computeBitwiseError(stateLabels, classifier, data, vis);
         else
             bitwiseError = sum(sum(abs(data - vis))) / (numCases * numVis);
         end
         
         % COMPUTE KL Divergence
         % =====================
         % klVector = computeKLDistance(distribution, q);
         
         % LOG the Learning Analytics
         % ==========================
         if useClass           
             fprintf(fid,'EPOCH: %d,\tLEARNING RATE: %5.6f\tNUM INVALID CODES: %d,\tERROR: %5.6f,\tTIME: %d mins\n', epoch, learnRate, numBad, bitwiseError, relativeTime);
         else
             fprintf(fid,'EPOCH: %d,\tLEARNING RATE: %5.6f\tERROR: %5.6f,\tTIME: %d mins\n', epoch, learnRate, bitwiseError, relativeTime);
             fprintf(fid,'DELTA: WEIGHTS, %3.4f\t VIS BIAS, %3.4f\t HID BIAS, %3.4f\t\n\n', delta_visToHid , delta_visBias, delta_hidBias);
         end
                           
         % DISPLAY Model Output
         % =====================
         % if meaningful (ex. MNIST)
%          close();
%          
%          if size(vis,1) >= 100
%             displayCases(vis(1:100,:),false);
%          else
%             displayCases(vis,false);
%          end
         
                   
         % GENERATE Plot of MNIST Generated Samples From the Model
         % =======================================================
         
%         title(sprintf('First 100 Samples generated by the model after %d Training epochs', epoch));

%          if epoch == maxEpoch             
%              outputFilename = strcat('outputSamples_',num2str(currTime(2)),'_',num2str(currTime(3)),'_',num2str(currTime(4)),'_',num2str(currTime(5)));
%              print('-depsc', outputFilename);
%          end
          
         count = count + 1;         
         
         old_visToHid = modelWeights{1};
         old_hidBias = modelBiases{2};
         old_visBias = modelBiases{1};

    end
     
    
    %%%%%%%%%%%%
    % LEARNING
    %%%%%%%%%%%%
    
    for batch = 1:numBatches
                
        dataBatch = batchdata{batch}; 
        vis = dataBatch(:,indices{1});
        
        %%%%%%%%%%%%%%%%
        % GIBBS SAMPLING
        %%%%%%%%%%%%%%%%%
                       
        visToHid = modelWeights{1};
        hidBias = repmat(modelBiases{2},batchSize,1);
        visBias = repmat(modelBiases{1},batchSize,1);
        
        % compute hiddens
        probs = 1./(1 + exp(- vis * visToHid - hidBias));
        if stochastic
            hid = probs > rand(batchSize, numHid);
        else
            hid = probs;
        end
        
        samplesConditional(:,indices{1}) = vis;
        samplesConditional(:,indices{2}) = hid;
        
        for it = 1:numGibbs
            % compute visibles
            probs = 1./(1 + exp(- hid * visToHid' - visBias));
            if stochastic
                vis = probs > rand(batchSize, numVis);
            else
                vis = probs;
            end
            
            % compute hiddens
            probs = 1./(1 + exp(- vis * visToHid - hidBias));
            if stochastic
                hid = probs > rand(batchSize, numHid);
            else
                hid = probs;
            end
        end

        samplesJoint(:,indices{1}) = vis;
        samplesJoint(:,indices{2}) = hid;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END GIBBS
            
        
        %%%%%%%%%%%%%%
        % UPDATE MODEL
        %%%%%%%%%%%%%%%             
        
        
        prodConditionalVH = samplesConditional(:,indices{1})' * samplesConditional(:,indices{2}) / batchSize;
        prodConditionalVBias = sum(samplesConditional(:,indices{1})) / batchSize;
        prodConditionalHBias = sum(samplesConditional(:,indices{2})) / batchSize;

        prodJointVH = samplesJoint(:,indices{1})' * samplesJoint(:,indices{2}) / batchSize;
        prodJointVBias = sum(samplesJoint(:,indices{1})) / batchSize;
        prodJointHBias = sum(samplesJoint(:,indices{2})) / batchSize;

        VH_Update = momentum * VH_Update + (prodConditionalVH - prodJointVH) * learnRate;
        VBias_Update = momentum * VBias_Update + (prodConditionalVBias - prodJointVBias) * learnRate;
        HBias_Update = momentum * HBias_Update + (prodConditionalHBias - prodJointHBias) * learnRate;

        modelWeights{1} = modelWeights{1} + VH_Update;
        modelBiases{1} = modelBiases{1} + VBias_Update;
        modelBiases{2} = modelBiases{2} + HBias_Update;
               
        %%%% END MODEL UPDATE
        
    end
    
end


% G_old = ones(numVis, numHid);
% G = ones(numVis, numHid);
% localUpdateGains = G_old .* G > 0;
% 
% G_old = G;
% G = G.*(~localUpdateGains * 0.95 + localUpdateGains + 0.5 * localUpdateGains);


