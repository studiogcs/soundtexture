function [target_S] = run_stats(P)
%
% [SYNTH_SOUND] = RUN_SYNTHESIS(P)
%
% Generates a synthetic signal that matches the statistics of a WAV file
% whose name is specified in P.ORIG_SOUND_FILENAME.
%
% Must be passed a set of parameters in P that can be set to their default
% values by the SYNTHESIS_PARAMETERS script.
%
% Please see the README_sound_texture file for more information.
%
%

% This code is part of an instantiation of a sound texture synthesis
% algorithm developed with Eero Simoncelli and described in this paper:
%
% McDermott, J.H., Simoncelli, E.P. (2011) Sound texture perception via
% statistics of the auditory periphery: Evidence from sound synthesis.
% Neuron, 71, 926-940. 
%
% Dec 2012 -- Josh McDermott <jhm@mit.edu>
%
% modified Nov 2013 to allow initialization with arbitrary stored waveform.
% Nov 2013 -- Josh McDermott <jhm@mit.edu>


if P.avg_stat_option==0
    fprintf('Starting %s\n\n',P.orig_sound_filename);
elseif P.avg_stat_option==1
    fprintf('Starting %s\n\n',[P.avg_filename ' - AVERAGE']);
elseif P.avg_stat_option==2
    fprintf('Starting %s\n\n',[P.avg_filename ' - MORPH with ratio of ' num2str(P.morph_ratio)]);
end
tic

%COMPUTE STATISTICS FROM SAMPLE
if P.avg_stat_option==0
    orig_sound = format_orig_sound(P);
    measurement_win = set_measurement_window(length(orig_sound),P.measurement_windowing,P);
    target_S = measure_texture_stats(orig_sound,P,measurement_win);
    target_S = edit_measured_stats(target_S,P);
    %generate subbands and envelopes of original sound for future use
    [orig_subbands,orig_subband_envs] = generate_subbands_and_envs(orig_sound, P.audio_sr, ...
        P.env_sr, P.N_audio_channels, P.low_audio_f, P.hi_audio_f, P.lin_or_log_filters, ...
        P.use_more_audio_filters,P.compression_option,P.comp_exponent,P.log_constant);
elseif P.avg_stat_option>=1
    [target_S, waveform_avg] = measure_texture_stats_avg(P);
    target_S = edit_measured_stats(target_S,P);
    [orig_subbands,orig_subband_envs] = generate_subbands_and_envs(waveform_avg, P.audio_sr, ...
        P.env_sr, P.N_audio_channels, P.low_audio_f, P.hi_audio_f, P.lin_or_log_filters, ...
        P.use_more_audio_filters,P.compression_option,P.comp_exponent,P.log_constant);
    orig_sound = waveform_avg;
    measurement_win = set_measurement_window(length(orig_sound),P.measurement_windowing,P);
end

