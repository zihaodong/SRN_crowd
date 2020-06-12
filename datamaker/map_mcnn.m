%MAP_MCNN   Translate .mat to density map.
%   map_mcnn(file_path) returns none
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
%     map_mcnn('img_path', 'UAVData/train_data/train_img', 'gt_path', 'UAVData/train_data/ground_truth', 'save_path', 'UAVData/train_data/train_gt', 'peoples', 1)

function map_mcnn(varargin)
    % Default parameter setting
    ip = inputParser;
    ip.addParameter('img_path', '../data/part_B_final/train_data/images');
    ip.addParameter('gt_path', '../data/part_B_final/train_data/ground_truth');
    ip.addParameter('save_path', '../data/part_B_final/train_data/train_gt');
    ip.addParameter('imsave_path', '../data/train_im');
    %ip.addParameter('scale', [-1, -1]);
    ip.addParameter('type', 'MCNN');
    ip.addParameter('peoples', 1);
    
    % Getopt
    ip.parse(varargin{:});
    results = ip.Results;
    img_path0 = results.img_path;
    gt_path0 = results.gt_path;
    save_path0 = results.save_path;
    imsave_path0 = results.imsave_path;
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
        imsave_path = sprintf('%s%s%s%s', imsave_path0, '/', name, '.jpg');
        
        % clear invalid coordinates
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
        gt(:,1) = ceil(gt(:,1) * j_scale);
        gt(:,2) = ceil(gt(:,2) * i_scale); 

        outputimage = uint8(zeros(720,  720));
        outputimage(1:size(I, 1), 1:size(I, 2), :) = outputimage(1:size(I, 1), 1:size(I, 2), :) + I;
        I = outputimage;
        
        % Generate 0 matrix and obtain a binary matrix by comparing the GT with the calibrated GT matrix
        D = zeros(size(I));
        for k =1:size(gt, 1)
          if gt(k,2)>0 && gt(k, 1)>0
              D(gt(k, 2), gt(k, 1)) = 1; % This matrix is the generated binary matrix 
          end
        end
        
        % human head counting 
        Headnum = 0;
        for m = 1:size(D, 1)
            for n = 1:size(D, 2)
                if D(m, n) == 1
                    Headnum = Headnum + 1;
                end
            end
        end
        
        % determine the radius of the head based on the distance 
        Distance = zeros(size(D, 1), size(D, 2)); % distance matrix 
        for p = 1:size(D,1)
            for q = 1:size(D, 2)
                if D(p, q)==1
                    for radius = 1:500
                    SearchHeads = searchhead(D, p, q, radius); % search radius increasing 
                    if size(SearchHeads, 1) > 4 % until 5 heads are found 
                       for r =1:size(SearchHeads, 1)
                           Distance(p, q) = Distance(p, q) + distance(p, q, SearchHeads(r, 1), SearchHeads(r, 2));
                       end
                       Distance(p, q) = (1 / (size(SearchHeads, 1) - 1)) * Distance(p, q);
                       break
                    end
                    end
                end
            end
        end
        
        m = size(I, 1);
        n = size(I, 2);
        d_map = zeros(ceil(m), ceil(n));
        
        % Generate density map 
        for j = 1 : size(gt, 1)
            ksize = Distance(gt(j, 2),(gt(j, 1)));

            ksize = max(ksize,10);
            ksize = min(ksize,80);
            ksize = ceil(ksize);
            radius = ceil(ksize/2);
            sigma = 0.3*ksize;
            x_ = max(1,min(n,floor(gt(j, 1))));
            y_ = max(1,min(m,floor(gt(j, 2))));
            h = fspecial('gaussian', ksize, sigma);

            hsize = size(h);
            hend = hsize(2);
            b = 0;

            if (x_ - radius + 1 < 1)
               for ra = 0 : radius - x_ -1
                   h(:, hend - ra) = h(:, hend - ra) + h(:, 1);
                   h(:, 1) = [];
               end
            end

            if (x_ + ksize - radius > n)
               for ra = 0 : x_ + ksize - radius -n-1
                   h(:, 1 + ra) = h(:, 1 + ra) + h(:, size(h,2));
                   h(:, size(h,2)) = [];
               end
            end

            if (y_ - radius + 1 < 1)
               for ra = 0 : radius - y_ -1
                   h(hend - ra,:) = h(hend - ra,:) + h(1,:);
                   h(1,:) = [];
               end
            end

            if (y_ + ksize -radius > m)
               for ra = 0 : y_ + ksize - radius - m-1
                  h(1 + ra, :) = h(1 + ra, :) + h(size(h,1), :);
                  h(size(h,1), :) = [];
               end
            end

            d_map(max(y_-radius+1,1):min(y_+ksize-radius,m),max(x_-radius+1,1):min(x_+ksize-radius,n))...
                 = d_map(max(y_-radius+1,1):min(y_+ksize-radius,m),max(x_-radius+1,1):min(x_+ksize-radius,n))...
                  + h;
        end
        
        d_map = d_map * peoples;
        
        % ploting
        %filename
        count = sum(sum(d_map)) % counting
        pcolor(d_map)
        shading flat;
        caxis([0,0.025]);
        colorbar
        axis ij
        display(['saved:', num2str(i)]);
        
        % data storage
        outputD_map=zeros(size(D, 1), size(D, 2));
        outputD_map(1:size(d_map, 1), 1:size(d_map, 2)) = outputD_map(1:size(d_map, 1), 1:size(d_map, 2)) + d_map;
        save(save_path, 'outputD_map');
        imwrite(image_out, imsave_path, 'jpg');
        
        clear d_map;
    end
end
