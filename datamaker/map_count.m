%MAP_MCNN   Translate .mat to point map.
%  
%   default parameters
%     img_path:  'UAVData/train_data/train_img'
%     gt_path:   'UAVData/train_data/ground_truth'
%     save_path: 'UAVData/train_data/train_gt'
%     scale:     [-1, -1] represent for not resize, eg. [480, 640] is height
%     -> 480 and width->640
%     type:     'MCNN' ('P2P')
%     peoples:  1 represent for 1 point is how many peoples
%
%   Folder arragement
%     UAVData
%     ├── test_data
%     │   ├── ground_truth
%     │   ├── test_gt
%     │   └── test_img
%     └── train_data
%         ├── ground_truth
%         │   ├── GT_IMG_1.mat
%         │   └── GT_IMG_2.mat
%         ├── train_gt
%         └── train_img
%             ├── IMG_1.jpg
%             └── IMG_2.jpg

%   Example:
%     map_count('img_path', 'UAVData/train_data/train_img', 'gt_path', 'UAVData/train_data/ground_truth', 'save_path', 'UAVData/train_data/train_gt', 'peoples', 1)

function map_count(varargin)
    % Default parameter setting
    ip = inputParser;
    ip.addParameter('img_path', '../data/part_B_final/train_data/images');
    ip.addParameter('gt_path', '../data/part_B_final/train_data/ground_truth');%density map
    ip.addParameter('save_path', '../data/part_B_final/train_data/train_count_gt');%point map
    %ip.addParameter('scale', [-1, -1]);
    ip.addParameter('type', 'MCNN');
    ip.addParameter('peoples', 1);
    
    % Getopt
    ip.parse(varargin{:});
    results = ip.Results;
    img_path0 = results.img_path;
    gt_path0 = results.gt_path;
    save_path0 = results.save_path;
    type = results.type;
    peoples = results.peoples;

    % Get image list to process
    file_path = dir([img_path0, '/*.jpg']);
    for i = 1 : length(file_path)
        % Image/Density Labels
        filename = strsplit(file_path(i).name, '.');
        name = char(filename(1));

        % The image/density label path and save path
        img_path = sprintf('%s%s%s%s', img_path0, '/', name, '.jpg');
        gt_path = sprintf('%s%s%s%s', gt_path0, '/GT_', name, '.mat');
        save_path = sprintf('%s%s%s%s', save_path0, '/', name, '.mat');
 
        % Read the image/density label
        image = imread(img_path);
        image_out = imresize(image, [720,720]);
        try 
            I = rgb2gray(image);
        catch
            I=image;
        end
        load(gt_path);
        location = image_info{1}.location; % human head coordinates
        gt = ceil(location);
        
        % clear invalid coordinates
        gtsize = size(gt, 1);
        tmpgt = gt; % intermediate variable，easy to operate
        numofdel = 0; % the number of deletions used to correct the index
        for j = 1 : gtsize
            if gt(j, 1)<=0 || gt(j, 2)<=0 || gt(j,1)>size(image ,2)...
                    || gt(j, 2)>size(image, 1)
                tmpgt(j - numofdel, :) = []; % index is needed to correct
                numofdel = numofdel + 1;
            end
        end
        gt = tmpgt;
       
        
        % resize, the uniform size of 720 X 720
        i_scale = 720 / size(I, 1);
        j_scale = 720 / size(I, 2); % the scaling ratio of rows and columns

        
        I = imresize(I, [720,720]);
        outputD_map = zeros(720,  720);
        
        gt(:,1) = ceil(gt(:,1) * j_scale);
        gt(:,2) = ceil(gt(:,2) * i_scale); 
        
        #generate the point map matrix
        for j = 1:size(gt(:,1),1)
            outputD_map(gt(j,2),gt(j,1))=1;
        end
        
        save(save_path, 'outputD_map');
    end
