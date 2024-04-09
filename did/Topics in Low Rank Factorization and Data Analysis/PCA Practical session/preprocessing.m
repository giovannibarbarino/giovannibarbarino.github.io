function X = preprocessing(n,m)
% Takes the images in the image dataset in the folder P and outputs a matrix X
% where the columns of X are the images
% If n,m are specified, it resizes the images to nxm images before inserting 
% them into the matrix X


P = './yaleB01/';
% P = './yaleB01_outlier/';
D = dir(fullfile(P,'*.pgm'));
X = [];
for i = 1:size(D,1)
    Xi = imread(fullfile(P,D(i).name));
    if nargin == 2  
        Xi = imresize(Xi,[n m]);
    end 
    X(:,i) = double(Xi(:))/255;
end





end
