function test_labels = classifier_2(trainfile, testfile, out_train, out_test)
%{
Author : Vaibhav Malpani 
KNN Classification
4-Fold Accuracy ~ 93.4%

Parameters Set:
    1. Number of Neighbors (k)
    2. Choice of Distance (Euclidean, Mahalanobis, Manhattan,..)
    3. Weighted Voting (1/d, 1/d^2,...)
    4. Feature Selection
        optimal feature set 1 - 2,3,5,6,7,13,14
        optimal feature set 2 - 6,3,5,13,7,11,14
%}

for numloop = 1 : 2
    % numloop = 1 --> classifies test data
    % numloop = 2 --> classifies training data
    if numloop == 2
        testfile = trainfile;
    end
    
    train_mat = txt2mat(trainfile);
    test_mat = txt2mat(testfile);
    
    c_train = load(train_mat);
    c_test = load(test_mat);
    fprintf('Loaded Data Successfully!\n\n');
    
    cv_train_labels = c_train.labels;
    cv_train_features = c_train.features_norm;
    cv_test_features = c_test.features_norm;
    cv_test_features_out = c_test.features_norm;    
    
    clear c_train;
    clear c_test;
    
    % Optimal Feature Subset
%     feature_cols = [2,3,5,6,7,13,14];
%     feature_cols = [6,3,5,13,7,11,14];
    feature_cols = [2,3,4,5,6,7,11,13];
    cv_train_features = cv_train_features(:,feature_cols);
    cv_test_features = cv_test_features(:,feature_cols);      
    
    % Normalize features to zero mean and unit variance
    for i = 1 : size(cv_train_features,2)
            cv_train_features(:,i) = (cv_train_features(:,i) - mean(cv_train_features(:,i)))./std(cv_train_features(:,i));
            cv_test_features(:,i) = (cv_test_features(:,i) - mean(cv_test_features(:,i)))./std(cv_test_features(:,i));
    end

    fprintf('Got Training Data Features!\n\n');
    
    % k --> sqrt(n) to avoid overfitting(high k) as well as noise(low k)
    k = 4;

    if size(cv_train_features, 2) ~= size(cv_test_features, 2);
        fprintf('Columns in Train: %d', size(cv_train_features, 2));
        fprintf('Columns in Test: %d', size(cv_test_features, 2));
        error('KNN: unequal length of features in train and test samples');
    end
    
    % generate random test_labels to avoid allocating memory at runtime
    test_labels = cell(size(cv_test_features,1),1);
    
    % compute euclidean distance and return the k smallest
    [d, idx] = pdist2(cv_train_features, cv_test_features, 'seuclidean', 'Smallest', k);
    
    % Weighted majority voting. Weight = 1/(distance^2)
    for t = 1:size(cv_test_features,1)
        [b,~,group] = unique(cv_train_labels(idx(:,t)));
        vote_mat = accumarray(group,(1./d(:,t)).^2);
        [~, ind] = max(vote_mat);
        test_labels(t) = b(ind);
    end
    
    % Normal Voting
%     for t = 1:size(cv_test_features,1)
%         res_tab = tabulate(cv_train_labels(idx(:,t)))
%         [~, i] = max(cell2mat(res_tab(:,2)));
%         test_labels{t} = res_tab{i,1};
%     end
    
    final_out = [test_labels num2cell(cv_test_features_out)];
    fmt = '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
    
    % Output the classified data into a text file (training or test)
    if numloop == 1
        f_out_test = fopen(out_test,'w');
        for i = 1 : size(final_out,1)
            fprintf(f_out_test, fmt, final_out{i,:});
        end
    elseif numloop == 2
        f_out_train = fopen(out_train,'w');
        for i = 1 : size(final_out,1)
            fprintf(f_out_train, fmt, final_out{i,:});
        end
    end
end
fprintf('KNN Classification Completed!\n\n')
fprintf('Please Check "trainfile_output_2.txt" and "testfile_output_2.txt"\n')
end