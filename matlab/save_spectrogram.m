function [] = save_spectrogram(varargin)

file_path = '../../wavefiles/';
save_path = '../figures/spectrograms/';

w_len = 512;    % window length
t = 10;    % length of displayed signal in seconds

if size(varargin) > 0
    file_path = varargin(1);
end

if isdir(file_path)
    if file_path(end) ~= '/'
        file_path = [file_path '/'];
    end
    d = dir([file_path '*.wav']);
    for i = 1:size(d)
        [x, fs] = audioread([file_path d(i).name]); 
        x = x(find(x ~= 0,1):end,1);
        x = x(1:t*fs); % take first t seconds of left channel only
        w = hanning(w_len);
        fig = figure(i);
        spectrogram(x, w, floor(w_len/2), w_len*2, fs);
        view(-90,90) 
        set(gca,'ydir','reverse')
        title(d(i).name(1:end-4));
        saveas(fig, [save_path d(i).name(1:end-4),'.png']);
    end
end

end