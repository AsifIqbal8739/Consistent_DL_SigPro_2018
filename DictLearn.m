function [Count] = DictLearn(Data,Dict,Dict_O,noIt,K,method,alpha)
global St; St = 1;  % first atom to optimize or not, 1 = yes, 2 = no
Count = zeros(1,noIt);
    for it = 1:noIt
        % Sparse coding stage
        if ~strcmpi(method,'A1') && ~strcmpi(method,'A2')
            X = omp(Dict'*Data,Dict'*Dict,K);        
        elseif it == 1
            X = pinv(Dict)*Data;
        end
        Dt = Dict;
    % Dict Update Stage 
        switch lower(method)
            case 'ksvd'
                [Dict,X] = Optimize_K_SVD(Data,Dict,X);            
            case 's1'
                a = alpha;
                [Dict,X] = Optimize_S1(Data,Dict,X,a);
            case 'a1'
                a = alpha;
                [Dict,X] = Optimize_S1Adapt1(Data,Dict,X,a);
            case 'a2'
                a = alpha;
                [Dict,X] = Optimize_S1Adapt2(Data,Dict,X,a);                 
            otherwise
                error('Invalid Learning Method Specified');
        end
        
        Count(1,it) = NumAtomRec(Dict,Dict_O);               % To find # of atoms recovered
        disp([method,' Iteration # ',num2str(it),' Atoms recovered = ',num2str(Count(1,it))])  
        if norm(Dict - Dt,'fro')/norm(Dict,'fro') < 0.001 % || isnan(X(1,1)) == 1
            Count(1,it:end) = Count(1,it);
            break;
        end
        Dict = I_clearDictionary(Dict,X,Data);
        
    end
end

% K-SVD Dictionary Update Stage
function [D,W] = Optimize_K_SVD(Y,D,W)
    global St;
    Ek = Y - D * W;
    for k = St:size(D,2)
        I = find(W(k,:));
            if isempty(I); 
                disp('ola'); D(:,k) = rep_atom(Ek); continue;   
            end;
        Eki = Ek(:,I) + D(:,k)*W(k,I);
        [U,S,V] = svds(Eki,1,'L');
        % U is normalized
        D(:,k) = U;
        W(k,I) = S*V';
        Ek(:,I) = Eki - D(:,k)*W(k,I);
    end    
end

% S1 Dictionary Update Stage
function [D,W] = Optimize_S1(Y,D,W,alpha)
global St;
    Ek = Y - D * W;
    for k = St:size(D,2)       
        Eki = Ek + D(:,k)*W(k,:);
        for j = 1:2
            W(k,:) = sign(D(:,k)'*Eki).*max(abs(D(:,k)'*Eki)-alpha,0);
            D(:,k) = Eki * W(k,:)'/norm(Eki * W(k,:)');
        end
        Ek = Eki - D(:,k)*W(k,:);
    end
end

% S1Adaptive Dictionary Update Stage   MODE 1 SVD
function [D,W] = Optimize_S1Adapt1(Y,D,W,alpha)
global St;
Ek = Y - D * W;
for k = St:size(D,2)       
    Eki = Ek + D(:,k)*W(k,:);
    if nnz(W(k,:)) < 1 
        disp('ola'); D(:,k) = normc(rep_atom(Eki));
        W(k,:) = D(:,k)'*Eki;
        continue;   
    end;
    [u,s,v] = svds(Eki,1);
    D(:,k) = u;     W(k,:) = s.*v';
    alpha_ = alpha./abs(W(k,:)+0.001);
    % Alphas could be infinite due to division by zero, tackling it
%     alpha_(isinf(alpha_)) = alpha;
    for j = 1:1
        W(k,:) = sign(W(k,:)).*max(abs(W(k,:))-alpha_,0);
        tempp = ~isfinite(W(k,:));
        W(k,tempp) = 0;
        d = normc(Eki * W(k,:)');
        if nnz(diff(d) == 0)
            D(:,k) = normc(rep_atom(Eki));
        else
            D(:,k) = d;
        end
    end
    tempp = ~isfinite(W(k,:));
    W(k,tempp) = 0;
    Ek = Eki - D(:,k)*W(k,:);
end
end

% S1Adaptive Dictionary Update Stage   MODE 2 Power Iterations
function [D,W] = Optimize_S1Adapt2(Y,D,W,alpha)
global St;
Ek = Y - D * W;
for k = St:size(D,2)       
    Eki = Ek + D(:,k)*W(k,:); 
    if nnz(W(k,:)) < 1 
        disp('ola'); D(:,k) = normc(rep_atom(Eki)); 
        W(k,:) = D(:,k)'*Eki;
        continue;   
    end;
    % Power Iterations
    for i = 1:3
        W(k,:) = D(:,k)'*Eki;   D(:,k) = normc(Eki*W(k,:)');
    end
    alpha_ = alpha./abs(W(k,:)+0.001);
    % Alphas could be infinite due to division by zero, tackling it
    alpha_(isinf(alpha_)) = alpha;
    for j = 1:1
        W(k,:) = sign(W(k,:)).*max(abs(W(k,:))-alpha_,0);
        tempp = ~isfinite(W(k,:));
        W(k,tempp) = 0;
        d = normc(Eki * W(k,:)');
        if nnz(diff(d) == 0)
            D(:,k) = normc(rep_atom(Eki));
        else
            D(:,k) = d;
        end
    end
    Ek = Eki - D(:,k)*W(k,:);

%     if nnz(isnan(D(:,k))) ~= 0
%         D(:,k); 
%     end
end
end

%% Replace atom which isn't used
function [d] = rep_atom(Y)
    temp = sqrt(diag(Y'*Y));    % Calculating norm2 of all vectors in Y
    [~,b] = sort(temp,1,'descend');
    d = normc(Y(:,b(1)));
end

%% Function to Find # of Recovered Atoms
function [Count] = NumAtomRec(D_,D)   % Learnt and Original Dic
Count = 0;
for i = 1:size(D,2)
    Count = Count + (abs(1 - max(abs(D(:,i)'*D_))) <= 0.01);
end
Count = 100*Count/size(D_,2);
end

%% Function to replace unused / similar dictionary atoms
function Dictionary = I_clearDictionary(Dictionary,W,Data,st)
if ~exist('st')  % to show the dict convergence or not.
    st = 1;
end
T2 = 0.99;
T1 = 3;
K = size(Dictionary,2);
Er = sum((Data-Dictionary*W).^2,1); % remove identical atoms
G = Dictionary'*Dictionary; G = G-diag(diag(G));
for jj = st:1:K
    if max(G(jj,:)) > T2 || nnz(W(jj,:)) <= T1
        [~,pos] = max(Er);
        Er(pos(1)) = 0;
        Dictionary(:,jj) = Data(:,pos(1))/norm(Data(:,pos(1)));
        G = Dictionary'*Dictionary; G = G-diag(diag(G));
    end
end
end