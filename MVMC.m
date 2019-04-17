function [outY, theta] = MVMC(singleTrainFeaL, singleTrainFeaU, singleTestFea, ...
    trainLabelsL, trainLabelsU, testLabels, set, option, para)
% -------------------------------------------------------------------------
% Multiview matrix completion
% -------------------------------------------------------------------------

if ~option.bivTheta && ~option.uniformTheta && ~option.selfDefineTheta
    % ---------------------------------------------------------------------
    % Split the labeled data into two parts
    % ---------------------------------------------------------------------
    index1 = 1:set.nbPL; index2 = set.nbPL+1:set.nbL;
    
    W1 = cell(set.nbV,1);
    W2 = cell(set.nbV,1);
    if option.completeOnlyU
        Y_size = size([trainLabelsL trainLabelsU]);
        for v = 1:set.nbV
            tmpW = [nan(Y_size); singleTrainFeaL{v} singleTrainFeaU{v}];
            W1{v} = tmpW; W1{v}(1:set.nbP,index1) = trainLabelsL(:,index1);
            W2{v} = tmpW; W2{v}(1:set.nbP,index2) = trainLabelsL(:,index2); clear tmpW
        end
    end
    if option.completeOnlyT
        Y_size = size([trainLabelsL testLabels]); set.nbU = 0;
        for v = 1:set.nbV
            tmpW = [nan(Y_size); singleTrainFeaL{v} singleTestFea{v}];
            W1{v} = tmpW; W1{v}(1:set.nbP,index1) = trainLabelsL(:,index1);
            W2{v} = tmpW; W2{v}(1:set.nbP,index2) = trainLabelsL(:,index2); clear tmpW
        end
    end
    if ~option.completeOnlyU && ~option.completeOnlyT
        Y_size = size([trainLabelsL trainLabelsU testLabels]);
        for v = 1:set.nbV
            tmpW = [nan(Y_size); singleTrainFeaL{v} singleTrainFeaU{v} singleTestFea{v}];
            W1{v} = tmpW; W1{v}(1:set.nbP,index1) = trainLabelsL(:,index1);
            W2{v} = tmpW; W2{v}(1:set.nbP,index2) = trainLabelsL(:,index2); clear tmpW
        end
    end
    
    % ---------------------------------------------------------------------
    % Generate the data for learning the multiview combination coefficient THETA
    % ---------------------------------------------------------------------
    if option.predEntryLExist
        fprintf('Train: Loading completed single matrix ... ');
        Y12 = load('predEntryL.mat');
        Y12 = struct2cell(Y12);
        Y12 = Y12{1};
    else
        Y12 = cell(set.nbV,1);
        fprintf('Train: Completing single matrix ... ');
        option.status = 'train';
        for v = 1:set.nbV
            fprintf('%d: p1 ', v);
            if option.preLambda_trn
                para.lambda_trn = para.preLambda_trn(v);
            else
                para.lambda_trn = para.lambda;
            end
            % -------------------------------------------------------------
            % Complete the entries in part 2 using labeled data in part 1
            % -------------------------------------------------------------
            [Z, err] = MC(W1{v}, trainLabelsU, testLabels, set, option, para);
            Y2 = Z(1:set.nbP,index2); ERR2.N(v) = err.N; ERR2.X(v) = err.X; ERR2.Y(v) = err.Y; clear Z err
            fprintf('p2 ');
            % -------------------------------------------------------------
            % Complete the entries in part 1 using labeled data in part 2
            % -------------------------------------------------------------
            [Z, err] = MC(W2{v}, trainLabelsU, testLabels, set, option, para);
            Y1 = Z(1:set.nbP,index1); ERR1.N(v) = err.N; ERR1.X(v) = err.X; ERR1.Y(v) = err.Y; clear Z err
            % -------------------------------------------------------------
            % Combine the predicted entries in part 1 and part 2
            % -------------------------------------------------------------
            Y12{v} = zeros(set.nbP, set.nbL);
            Y12{v}(:,index1) = Y1; Y12{v}(:,index2) = Y2; clear Y1 Y2
        end
        if option.savePredEntryL
            save('predEntryL.mat', 'Y12', '-v7.3');
        end
    end
    clear W1 W2 index1 index2
    fprintf('Finished!\n');
    
    % ---------------------------------------------------------------------
    % Initialization of the multiview combination coefficients THETA
    % ---------------------------------------------------------------------
    randomInit = 0;
    if option.bivTheta || option.uniformTheta || option.selfDefineTheta
        theta_ini = para.theta;
    else
        if randomInit
            rand('seed', 123);
            theta_ini = rand(set.nbV, 1);
            theta_ini = theta_ini / sum(theta_ini(:));
        end
        theta_ini = (1.0 / set.nbV) * ones(set.nbV, 1);
    end
    
    % ---------------------------------------------------------------------
    % Optimize the multiview combination coefficients THETA
    % ---------------------------------------------------------------------
    switch option.loss
        case 'ls'
            theta = optimizeThetaLS(Y12, trainLabelsL, theta_ini, set, para);
        case 'svm'
            theta = optimizeThetaSVM(Y12, trainLabelsL, theta_ini, set, para);
        case 'map'
            theta = optimizeThetaMAP(Y12, trainLabelsL, theta_ini, set, para);
    end
else
    theta = para.theta;
end

% -------------------------------------------------------------------------
% Re-complete the matrix using all the labeled data and combine the output using the learned theta
% -------------------------------------------------------------------------

% ---------------------------------------------------------------------
% Initialization
% ---------------------------------------------------------------------
W = cell(set.nbV,1);
if option.bivTheta
    rangeV = para.viewInd:para.viewInd;
else
    rangeV = 1:set.nbV;
end
if option.completeOnlyU
    Y_size = size([trainLabelsL trainLabelsU]); YU_size = size(trainLabelsU);
    for v = rangeV
        W{v} = [trainLabelsL nan(YU_size); singleTrainFeaL{v} singleTrainFeaU{v}];
    end
end
if option.completeOnlyT
    Y_size = size([trainLabelsL testLabels]); YU_size = size(testLabels); set.nbU = 0;
    for v = rangeV
        W{v} = [trainLabelsL nan(YU_size); singleTrainFeaL{v} singleTestFea{v}];
    end
end
if ~option.completeOnlyU && ~option.completeOnlyT
    Y_size = size([trainLabelsL trainLabelsU testLabels]); YU_size = size([trainLabelsU testLabels]);
    for v = rangeV
        W{v} = [trainLabelsL nan(YU_size); singleTrainFeaL{v} singleTrainFeaU{v} singleTestFea{v}];
    end
end

% ---------------------------------------------------------------------
% Re-complete the matrices
% ---------------------------------------------------------------------
if option.predEntryUExist
    fprintf('Predict: Loading completed single matrix ... ');
    Y = load('predEntryU.mat');
    Y = struct2cell(Y);
    Y = Y{1};
else
    Y = cell(set.nbV,1);
    fprintf('Predict: Completing single matrix ... ');
    option.status = 'predict';
    for v = rangeV
        fprintf('%d ', v);
        if option.preLambda_tst
            para.lambda_tst = para.preLambda_tst(v);
        else
            para.lambda_tst = para.lambda;
        end
        [Z, err] = MC(W{v}, trainLabelsU, testLabels, set, option, para);
        Y{v} = Z(1:set.nbP,:); ERR.N(v) = err.N; ERR.X(v) = err.X; ERR.Y(v) = err.Y; clear Z err
    end
    if option.savePredEntryU
        save('predEntryU.mat', 'Y', '-v7.3');
    end
end
fprintf('Finished!\n');

% ---------------------------------------------------------------------
% Combine the outputs
% ---------------------------------------------------------------------
outY = zeros(Y_size);
for v = rangeV
    outY = outY + theta(v)*Y{v};
end

end

