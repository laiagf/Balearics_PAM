function[specs]= getSpectrograms(filepath, t, n, x, y, out_path)
% Function to get n t-second spectrograms from an audio file.
% Inputs:
%   - filepath: path of the .wav file (with file name)
%   - t: length in seconds of the each chunk that will be converted to a spectrogram
%   - n: number of expected spectrograms
%   - x, y: length and height of the desired spectrogram images
%   - out_path: path to store generate spectrograms. If false they wont be
%   stpred. Default:false
% Outputs:
%   - specs: matrix with all resized spectrograms

    [w, fs] = audioread(filepath);
    
    %check that the audio file is not faulted: it is not all 0s and it is long enough
    %if so, return 0
    if (max(w)==0 | length(w)/fs <(t*(n-1))) 
        specs = 0;
    else
        l = t*fs; %number of samples of each audio segment that will be converted to spectrogram
        a = zeros(x, y); %initialize matrix that will store resized spectrograms

        for (i = 1:(n-1)) %create spectrograms for the first n-1 chunks
            ini = 1 + (i-1)*l; %start sample
            fin = ini+l-1;%end sample
            chunk = w(ini:fin); 
            [original ,spec] =getSpec(chunk, fs); %get original and resized spectrograms
            if spec==-1 %check that no error occured
                spec = zeros(x, y);
            %disp([size(a) size(spec)])
            elseif not(nargin < 6)
                [path, name, ext] = fileparts(filepath)
                outfilename = out_path+name+"_"+num2str(i)
                imwrite(spec, outfilename+".png")
                imwrite(original, outfilename+"original.png")
            end

            a = cat(3,a, spec); %append resized spectrogram to matrix
        end

        %for last spectrogram (treated sepparately in case the audio is a few samples short)
        fin = length(w); %end of the chunk is end of the audio
        ini = fin -l+1; %make it of length l for consistency with other chunks
        %repeat previous procedure
        chunk = w(ini:fin);
        [original ,spec] =getSpec(chunk, fs);
        if spec==-1
            spec = zeros(x, y);
        %disp([size(a) size(spec)])
        end
        a = cat(3,a, spec);       
        specs = a(:, :, 2:(n+1)); %return all resized spectrograms 
    end 
    
