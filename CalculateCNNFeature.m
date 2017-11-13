function CalculateCNNFeature()
%==========================================================================
% usage: calculate the CNN feature given the image directory
%
% inputs
% rt_img_dir    -image database root path
% rt_data_dir   -feature database root path
%
% outputs
% database      -directory for the calculated sift features
%
% written by Liangke Gui
% modified by Hongliang Yu
% Jun. 2017, CMU
%==========================================================================

% This script is used to generate the DL featuers for the entire image
% Generate the fc6 feature
% used for learning unsupervised PBC

% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

disp('Extracting CNN features...');
addpath '/usr0/home/hongliay/caffe/matlab/'
%% configuration for caffe
% model files
model_def_file = '/usr0/home/hongliay/FeatureExtraction/deep-residual-networks/prototxt/ResNet-152-deploy.prototxt';
model_file = '/usr0/home/hongliay/FeatureExtraction/deep-residual-networks/models/ResNet-152-model.caffemodel';

% set to use GPU or CPU
caffe.set_mode_gpu();
caffe.set_device(3);

global net
net = caffe.Net(model_def_file,model_file,'test');

imglist_file = '/multicomp/users/hongliay/datasets/MPII/jpgs_2050.txt';
data_dir = '/multicomp/users/hongliay/datasets/MPII/features_jpg';
img_dir = '/multicomp/users/hongliay/datasets/MPII/jpg';

Extract_Features(img_dir, data_dir, imglist_file)
end

function Extract_Features(rt_img_dir, rt_data_dir, imglist_file)
global net
paths = textread(imglist_file, '%s');
for i = 1:length(paths)
    feats = [];
    img = imread(fullfile(rt_img_dir,paths{i}));
    fpath = fullfile(rt_data_dir,strrep(paths{i},'.jpg','.mat'));  
    [pathstr,~,~] = fileparts(fpath);
    if ~isdir(pathstr)
        mkdir(pathstr);
    end
    
    fprintf('Processing %d of %d\n', i,length(paths));
    input_data = {prepare_image(img)};
    % do forward pass to get scores
    % scores are now Width * Height * Channels * Num
    scores = net.forward(input_data);
    
    feats.fc1000 = net.blobs('fc1000').get_data();
    
    save(fpath, 'feats');
end

end

% ------------------------------------------------------------------------
function image = prepare_image(im)
% ------------------------------------------------------------------------
mean_pix = [103.939, 116.779, 123.68];
IMAGE_DIM = 256;
CROPPED_DIM = 224;

% resize to fixed input size
im = single(im);

if size(im, 1) < size(im, 2)
    im = imresize(im, [IMAGE_DIM NaN],'bilinear');
else
    im = imresize(im, [NaN IMAGE_DIM],'bilinear');
end
if size(im,3) == 1
    im = cat(3,im,im,im);
end
% RGB -> BGR
im = im(:, :, [3 2 1]);

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');

indices_y = [0 size(im,1)-CROPPED_DIM] + 1;
indices_x = [0 size(im,2)-CROPPED_DIM] + 1;
center_y = floor(indices_y(2) / 2)+1;
center_x = floor(indices_x(2) / 2)+1;

curr = 1;
for i = indices_y
    for j = indices_x
        images(:, :, :, curr) = ...
            permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
        images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
        curr = curr + 1;
    end
end
images(:,:,:,5) = ...
    permute(im(center_y:center_y+CROPPED_DIM-1,center_x:center_x+CROPPED_DIM-1,:), ...
    [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);

% mean BGR pixel subtraction
for c = 1:3
    images(:, :, c, :) = images(:, :, c, :) - mean_pix(c);
end

image = images(:, :, :, 5);
end
