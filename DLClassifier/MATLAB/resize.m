function [img_mp] = resize(img, x, y)
%Function to resize an image to a specific size (length x, height y) by
%maxpooling. It divides the image into bins of equal size, distributing the
%extra columns in a way that they are linearly spaced.
%Outputs:
%   - img_mp: matrix containing downsampled image
    
    img_mp = zeros(y,x); %initialize matrix for downsampled image
    
    len=size(img, 2); %length of original image
    ht  = size(img, 1); %height of original image
    
    ly=floor(ht/y); %minimum height of bin (number of rows in original image contributing to a single row in new image)
    lx = floor(len/x); %minimum length of bin


    remainder_x = (len/x-lx)*x; %unused columns in original image if each column of the new image has only been sampled from lx columns 
    remainder_y = (ht/y-ly)*y; %unused rows
    

    inds_x_extra_sample = round(linspace(1,x, remainder_x)); %positions where the extra columns will be redistributed
    inds_y_extra_sample = round(linspace(1, y, remainder_y)); %positions where the extra rows will be redistributed


    inds_x = zeros(x, 1)+lx; %starting indices if all bins were of length lx
    inds_x(inds_x_extra_sample) =lx+1; %add the extra columns
    fins_x = cumsum(inds_x); %the cummulative sum of lengths marks the end of each horizontal bin
    inis_x = [1, (fins_x(1:x-1)+1)']; %start indices of horizontal bins

    %repeat for rows:
    inds_y = zeros(y, 1)+ly;
    inds_y(inds_y_extra_sample) =ly+1; %add the extra columns
    fins_y = cumsum(inds_y); %the cummulative sum of lengths marks the end of each horizontal bin
    inis_y = [1, (fins_y(1:y-1)+1)']; %start indices of horizontal bins
    
    
    %iterate over rows and columns of new image
    for i=1:x
        for j=1:y
            ini_x = inis_x(i); %find the starting horizontal index of bin 
            fin_x = fins_x(i); %end horizontal index 
            ini_y = inis_y(j); %starting vertical index of bin
            fin_y = fins_y(j); %end vertical index of bin
            clip = img(ini_y:fin_y, ini_x:fin_x); %crop bin that will be downsampled to one pixel
            
            img_mp(j, i) = max(max(clip)); %Find max value of the bin and store it in new matrix
            
        end
    end
    


