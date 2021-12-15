%% setup
clear; clc;
addpath(genpath('code_imdistort'));


%% read the info of pristine images

tb = readtable('kadis700k_ref_imgs.csv');
tb = table2cell(tb);

%% generate distorted images in dist_imgs folder
ref_img_name = {};
d_type = [];
d_level = [];
dis_img_name = {};
label = [];
cnt = 1;

for i = 1:size(tb,1)
    
%     if ~exist(['ref_imgs/' tb{i,1}])
%         disp(['ref_imgs/' tb{i,1}]);
%         break;
%     end
    
    if tb{i,2} == 13 || tb{i,2} == 23
%         cnt = cnt + 1;
        continue;
    end
    
    ref_im = imread(['ref_imgs/' tb{i,1}]);
    dist_type = tb{i,2};
    
    for dist_level = 1:5
        [dist_im] = imdist_generator(ref_im, dist_type, dist_level);
        strs = split(tb{i,1},'.');
        dist_im_name = [strs{1}  '_' num2str(tb{i,2},'%02d')  '_' num2str(dist_level,'%02d') '.jpg'];
        disp(dist_im_name);
        % imwrite(dist_im, ['dist_imgs/' dist_im_name]);
        imwrite(dist_im, ['dist_imgs/' dist_im_name], 'quality', 100);
        
        img1 = double(rgb2gray(ref_im));
        img2 = double(rgb2gray(dist_im));
        score = msssim(img1, img2);
        
        ref_img_name{cnt, 1} = tb{i,1};
        d_type(cnt, 1) = dist_type;
        d_level(cnt, 1) = dist_level;
        dis_img_name{cnt, 1} = dist_im_name;
        label(cnt, 1) = score;
        cnt = cnt + 1;
    end
    
end

save('kadis700k.mat', 'ref_img_name', 'd_type', 'd_level', 'dis_img_name', 'label');

a = 1;





