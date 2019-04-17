function [theta_new, obj_new, obj] = optimizeThetaLS(Z, trainLabelsL, theta, set, para)
% ---------------------------------------------------------------------
% Optimization of the multiview combination coefficients THETA with the least square loss
% ---------------------------------------------------------------------

omega_z = find(~isnan(Z{1}));
omega_z_len = length(omega_z);
Y = trainLabelsL; clear trainLabelsL
Y(Y == -1) = 0;
% for v = 1:set.nbV
%     Z{v}(Z{v} < 0) = 0;
% end
% --------------------------------------------
% Compute the quadratic term
% --------------------------------------------
H = zeros(set.nbV, set.nbV);
for i = 1:set.nbV
    for j = 1:set.nbV
        H(i,j) = 1.0/omega_z_len * trace(Z{i}'*Z{j});
    end
end

% --------------------------------------------
% Compute the linear term
% --------------------------------------------
h = zeros(set.nbV, 1);
for i = 1:set.nbV
    h(i) = 1.0/omega_z_len * trace(Z{i}'*Y);
end

% --------------------------------------------
% Update \theta by adopting coordinate descent
% --------------------------------------------
obj = 0.5*((theta'*H*theta) - 2*(h'*theta) + para.eta*(theta'*theta)) + ...
    0.5/omega_z_len*trace(Y'*Y);
theta_new = coorDesTheta(H, h, theta, set, para);
obj_new = 0.5*((theta_new'*H*theta_new) - 2*(h'*theta_new) + para.eta*(theta_new'*theta_new)) ...
    + 0.5/omega_z_len*trace(Y'*Y);

end

