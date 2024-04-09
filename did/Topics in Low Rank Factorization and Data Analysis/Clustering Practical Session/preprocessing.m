function Y = preprocessing(d)
% Takes the images in ATT and put them as columns of a matrix X
% then apply PCA to X with dimension d and outputs the 
% coefficient matrix Y


P = './ATT/';
N = 20; X = [];
for j = 1:N
    Paux = [P, 's', num2str(j), '/'];
    D = dir(fullfile(Paux,'*.pgm'));
    for i = 1:size(D,1)
        Xi = imread(fullfile(Paux,D(i).name));
        X = [ X double(Xi(:))/255];
    end
end
[~,~,Y] = faces_pca(X,d);
% Y = S*V'; 

end
