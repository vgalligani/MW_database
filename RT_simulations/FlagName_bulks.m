% FORMAT   fname = FlagName(flag)
%
% Provides a char name according to the given flag.
% The flag specifies the type of simulation.
%
% OUT   fname  Char name according to flag
% IN    flag   Flag specifying the type of simulat.
%              = 0 => All hydrom; no cloud liquid water (clw)
%              = 1 => Snow only; no clw
%              = 2 => Ice only; no clw
%              = 3 => Rain only; no clw
%              = 4 => Clear sky; no clw
%
% 20.05.2020 Vasileios Barlakas
%
function [fname, graupel_habit_id] = FlagName(flag,graupel_SSD,hail_d)

if flag == 0
   fname = 'AllSky';
   graupel_habit_id = nan

elseif flag == 1
   fname = 'BulkSIMS_RainOnly_HR';
   graupel_habit_id = nan;

elseif flag == 2
   fname = 'BulkSIMS_RainOnly_LR';
   graupel_habit_id = nan

elseif flag == 3
   %fname = 'BulkSIMS_HailOnly';
   fname = strcat('BulkSIMS_HailOnly_HWC', string(hail_d)); % 0.2*5
   graupel_habit_id = nan

elseif flag == 4   
   fname = strcat('BulkSIMS_RWCHR_HWC', string(hail_d)); % 0.2*5
   graupel_habit_id = nan

elseif flag == 5
   fname = strcat('BulkSIMS_RWCLR_HWC', string(hail_d)); % 0.2*5
   graupel_habit_id = nan



end


