%% Collect stats on sound files
% Using McDermott's code

synthesis_parameters_demo;

files =   [{'Lathrop_Noisy_Short.wav'}, ...
                {'Applause_-_enthusiastic2.wav'}, ...
                {'Bubbling_water.wav'}, ...
                {'white_noise_5s.wav'}, ...
                {'Writing_with_pen_on_paper.wav'}];

for i = 1:length(files)
    
    P.orig_sound_filename = files{i};

    [stats(i)] = run_stats(P);

end

save('all_stats.mat','files','stats');