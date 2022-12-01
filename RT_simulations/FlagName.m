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
function fname = FlagName(flag)

if flag == 0
   fname = 'AllSky';
elseif flag == 1
   fname = 'SnowOnly';
elseif flag == 2
   fname = 'IceOnly';
elseif flag == 3
   fname = 'RainOnly';
elseif flag == 4
   fname = 'ClearSky';
end
