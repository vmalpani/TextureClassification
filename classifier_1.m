function classifier_1(trainfile, testfile, out_train, out_test)
%{
Author : Vaibhav Malpani 
Bayesian Classification
4-Fold Accuracy ~ 91.78%
Optimal feature set 1 - {3,5,6,7,13}
Optimal feature set 2 - {7,3,5,4,13}
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
    
%     Optimal Feature Subset
%     feature_cols = [3,5,6,7,13];
    feature_cols = [7,3,5,4,13];
    cv_train_features = cv_train_features(:,feature_cols);
    cv_test_features = cv_test_features(:,feature_cols);       

    % Normalize features to zero mean and unit variance
%     for i = 1 : size(cv_train_features,2)
%         cv_train_features(:,i) = (cv_train_features(:,i) - mean(cv_train_features(:,i)))./std(cv_train_features(:,i));
%         cv_test_features(:,i) = (cv_test_features(:,i) - mean(cv_test_features(:,i)))./std(cv_test_features(:,i));
%     end
    
    fprintf('Got Train Features!\n\n');
    
    if size(cv_train_features, 2) ~= size(cv_test_features, 2);
        fprintf('Columns in Train: %d', size(cv_train_features, 2));
        fprintf('Columns in Test: %d', size(cv_test_features, 2));
        error('Bayesian: unequal length of features in train and test samples');
    end
    
    % Find unique class labels and their corresponding indexes
    % Generate prior probability using the class frequency
    [uniVal,~,Idx] = unique(cv_train_labels);
    test_labels = cell(size(cv_test_features,1),1);
    class_freq = accumarray(Idx(:),1);
    prior_prob = class_freq./sum(class_freq);
    
    % Store mean and variance of each feature in an array of structures
    class_index = [];
    class_mean = [];
    class_covariance = [];
    final_out = [];
    for i = 1 : size(uniVal, 1)
        class_index = find(i == Idx);
        class_mean = mean(cv_train_features(class_index, :));
        class_covariance = var(cv_train_features(class_index, :));    
%         class_covariance = class_covariance + .0001 * eye(size(class_covariance,2));
        final_out(i).idx = class_index;
        final_out(i).mu = class_mean;
        final_out(i).sigma = class_covariance;
    end
    
    pxw = zeros(size(cv_test_features,1), size(uniVal,1));
    
%     row - p(x|w_i) likelihood for each observation
%     col - classes
%     p(w_i) - each row has priors of all classes
    for j = 1 : size(uniVal, 1)
        pxw(:,j) = mvnpdf(cv_test_features, final_out(j).mu, final_out(j).sigma);
    end
    
    % Find the class corresponding to maximum posterior probability
    pwx = pxw .* repmat(prior_prob', size(cv_test_features,1), 1);
    [maxVal maxInd] = max(pwx');
    
    for i  = 1 : size(maxInd, 2)
        test_labels(i) = uniVal(maxInd(i));
    end
    
    final_out = [test_labels num2cell(cv_test_features_out)];
    
    % Output the classified data into a text file (training or test)
    fmt = '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
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
fprintf('Bayesian Classification Completed!\n\n')
fprintf('Please Check "trainfile_output_1.txt" and "testfile_output_1.txt"\n')
fclose(f_out_test);
fclose(f_out_train);
end