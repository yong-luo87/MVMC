function theta_new = coorDesTheta(H, h, theta, set, para)

loop = 1; iter = 0;
obj = 0.5*(theta'*H*theta + para.eta*(theta'*theta)) - h'*theta;
theta_new = theta;
while loop
    iter = iter + 1;
    % --------------------------------------------
    % Randomly select two elments to update
    % --------------------------------------------
    %     [maxGrad, idxMax] = max(grad);
    %     [minGrad, idxMin] = min(grad);
    %     i = idxMax; j = idxMin;
    %     clear maxGrad idxMax minGrad idxMin
    rand('seed', iter);
    thetaPerm = randperm(set.nbV);
    for k = 1:1 % floor(set.nbV/2)
        i = thetaPerm(2*k-1); j = thetaPerm(2*k);
        % i = thetaPerm(1); j = thetaPerm(2);
        
        % --------------------------------------------
        % Update the selected two elments using coordinate descent
        % --------------------------------------------
        temp = H(i,i) - H(i,j) - H(j,i) + H(j,j);
        numer_i = para.eta*(theta(i)+theta(j)) + (h(i)-h(j)) + temp*theta(i) - (H(i,:)-H(j,:))*theta;
        numer_j = para.eta*(theta(i)+theta(j)) + (h(j)-h(i)) + temp*theta(j) - (H(j,:)-H(i,:))*theta;
        denom = temp + 2.0*para.eta; clear temp
        if numer_i <= 0
            theta_new(i) = 0; theta_new(j) = theta(i)+theta(j);
        end
        if numer_j <= 0
            theta_new(j) = 0; theta_new(i) = theta(i)+theta(j);
        end
        if numer_i > 0 && numer_j > 0
            theta_new(i) = numer_i / denom;
            theta_new(j) = theta(i) + theta(j) - theta_new(i);
        end
    end
    
    % --------------------------------------------
    % Check the convergence
    % --------------------------------------------
    obj_new = 0.5*(theta_new'*H*theta_new + para.eta*(theta_new'*theta_new)) - h'*theta_new;
    % grad_new = (H2 + 2*(para.gamma_B+eps)*eye(size(H2)))*Beta_new - h;
    if abs(obj - obj_new) < 1e-4 || iter >= 500
        loop = 0;
    else
        clear Beta
        theta = theta_new;
        obj = obj_new;
        % grad = grad_new;
    end
end

end

