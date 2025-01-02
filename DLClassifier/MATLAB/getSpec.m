function[croppedSpec, scaledCroppedSpec]= getSpec(w, fs, outputFs, fftLen, img_x, img_y)

% make raw matrix of a spectrogram data from audiofile, and downsample it
% to a specific seize. 
%
% Inputs:
%   - w: waveform
%   - fs: sampling ratio
%   - outputFs: new sampling ratio
%   - fftLen
%   - img_x: wanted length of resized spectrogram
%   - img_y: wanted height of resized spectrogram
%
% Outputs:
%   - croppedSpec: raw matrix of the spectrogram cropped to ???kHz and ??? kHz 
%   - scaledCroppedSpec: croppedSpec downsampled to a specific size (x,y) using
%   the resize custom function

if nargin < 3
    outputFS = 48000;
end
if nargin < 4
    fftLen = 512;
end
if nargin < 5
    img_x = 64;
end
if nargin <6
    img_y=64;
end

w=w-mean(w); %Remove offset 

rw = resample(w, outputFS, fs); %Resample all recordings to a frequency of 48kHz to center sperm whale clicks (especially important since recorders have different sampling ratios!)


spec = spectrogram(rw, hann(fftLen), fftLen/2, fftLen); %generate spectrogram

spec = 20*log10(abs(spec)); %take only the magnitude


croppedSpec  = spec(10:size(spec,1)-10, :); %Crop limit frequencies (to avoid weird stuff happening around the edges)

% Whiten the image by substracting off a median value:
medVal = median(croppedSpec, 2); % get the median level for each frequency
croppedSpec = croppedSpec-medVal; % subtract it off.


finiteValsCroppedSpec = croppedSpec(isfinite(croppedSpec)); %get only finite values of the spectrogram


if length(finiteValsCroppedSpec)==0 %Check that the file wasnt faulted. If it was, return -1
    croppedSpec=-1; 
    scaledCroppedSpec=-1;

else %If all was ok with the file, we can normalise 
    minValCropped = min(finiteValsCroppedSpec);
    maxValCropped = max(finiteValsCroppedSpec);
    croppedSpec = (croppedSpec-minValCropped(1))/(maxValCropped(1)-minValCropped(1));
    
    
    %downsample image to the desire final size and repeat normalisation for this matrix

    resized_spec = resize(croppedSpec, img_x, img_y);

    %Find finite values again
    finiteVals = resized_spec(isfinite(resized_spec));
    
    %Check if fault
    if length(finiteVals)== 0 
        scaledCroppedSpec =-1;
    else
        %normalize
        minVal = min(finiteVals);
        maxVal = max(finiteVals);
        scaledCroppedSpec = (resized_spec-minVal(1))/(maxVal(1)-minVal(1));
    end
    
end




