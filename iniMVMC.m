function [option, para] = iniMVMC( )
% -------------------------------------------------------------------------
% Initialization of the options and parameters.
% -------------------------------------------------------------------------

option.verbose = 2;
% ---------------------------------------------------
% Choosing the algorithm: 'MC_1', 'MC_Pos' or 'MC_Simplex'
% ---------------------------------------------------
option.alg = 'MC_1';

% ---------------------------------------------------
% Choosing the loss function for optimizing theta
% ---------------------------------------------------
option.loss = 'svm'; % 'ls', 'svm', or 'log'

% ---------------------------------------------------
% Choosing the labeled number or fraction of the data
% ---------------------------------------------------
para.num_frac = 100;

% ---------------------------------------------------
% Choosing the size of the validation set, fraction of
% the test set if in (0,1)
% ---------------------------------------------------
para.size_valid = 0.2;

% ---------------------------------------------------
% Choosing to only use the visual features or not
% ---------------------------------------------------
option.visualOnly = 0;

% ---------------------------------------------------
% Choosing to normalize the kernel matrices or not
% ---------------------------------------------------
option.normKer = 0;

% -----------------------------------------------
% Choosing the maximum iterations
% -----------------------------------------------
para.nbIterMax = 100;

%------------------------------------------------------
% Choosing the stopping criterion
%------------------------------------------------------
option.stopdiffobj = 0;         % use difference of objective value for stopping criterion
option.stopvariationtheta = 1;  % use variation of graph weights for stopping criterion

%------------------------------------------------------
% Choosing the stopping criterion value
%------------------------------------------------------
para.seuildiffobj = 1e-4;       % stopping criterion for objective value difference
para.seuildifftheta = 1e-4;     % stopping criterion for graph weight variation

% ---------------------------------------------------
% Choosing to randomly split training set into 
% labeled and unlabeled set or not
% ---------------------------------------------------
option.randSplitLU = 0;
option.dataSelectLU = 1;

% ---------------------------------------------------
% Choosing to complete only test data or both unlabeled and test data
% ---------------------------------------------------
option.completeOnlyU = 0;
option.completeOnlyT = 0;

if option.completeOnlyU
    option.completeOnlyT = 0;
end
if option.completeOnlyT
    option.completeOnlyU = 0;
end

% ---------------------------------------------------
% Choosing to perform KPCA projection or not
% ---------------------------------------------------
option.KPCAprj = 1;
option.KPCA_LU = 0;
option.KPCA_Tag = 0;
option.prjExist = 0;

% ---------------------------------------------------
% Setting the dimensionality after KPCA projection, ratio if in (0,1]
% ---------------------------------------------------
para.rDim = 50;

% ---------------------------------------------------
% Setting the stopping criterion value of single view matrix compeletion
% ---------------------------------------------------
para.seuildiffobjMC = 1e-2;

% ---------------------------------------------------
% Setting the factor and stopping value for deriving
% a decreasing sequence of mu
% ---------------------------------------------------
para.red_factor = 0.25;
para.mu_thresh = 1e-12;
para.mu = 1e-12;

% ---------------------------------------------------
% Setting the parameter of the generalized log loss
% ---------------------------------------------------
para.rou = 3;

% -----------------------------------------------
% Setting the multiview option and parameter
% -----------------------------------------------
option.uniformTheta = 0;
option.bivTheta = 0; para.feaInd = 7;
option.selfDefineTheta = 0;
para.eta = 1e-1;

% ---------------------------------------------------
% some algorithms parameters and options
% ---------------------------------------------------
para.lambda = 1e-3;

para.ratioPL = 0.5;
option.predEntryLExist = 0;
option.predEntryUExist = 0;
option.savePredEntryL = 1;
option.savePredEntryU = 1;

% ---------------------------------------------------
% Some options for MC_Pos and MC_Simplex
% ---------------------------------------------------
% Choosing to project to positive or not
option.posPrj = 0;
% Choosing to project to simplex or not
option.simPrj = 0;
% Choosing the loss function for features
option.shldNrm = 0;
% Choosing the label value type, 1 if -1/1 labels
option.bin_type = 1;

end

