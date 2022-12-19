% FORMAT   test_gmi(sat_alt,flag,clear_only)
%
%    Simple modeling framework that creates profiles for ARTS for both
%    clear and all-sky conditions, initiates the ARTS simulations, 
%    and stores any related output.
%
% OUT    *.mat      ARTS output stored in matlab format
% IN     flag       Hail/graupel/rain combination experiment selectetion
%                   Accodingly, the input/output filenames are set
%                   = 4 => Clear sky; no clw
%                   = 1 => Rain only (heavy rain); no clw 
%                   = 2 => Rain only (light rain); no clw
%                   = 3 => Graupel only (test 1))
%
%       Graupel_Species: test: GemGraupel, IconGraupel, SphericalGraupel
%       clear_only  Flag for clear- (1) or all-sky (0) simulations.
% 
% 19.03.2020 Vasileios Barlakas and Patrick Eriksson
% 01.01.2022 Adapted Vito Galligani
%
function [arts_tb, arts_cl] = test_gmi(flag, clear_only, graupel_SSD, hail_shape)

%addpath('/home/victoria.galligani/Work/Matlab/arts-sspdb-interface-main/DataInterfaces/Matlab/')
%addpath('/home/victoria.galligani/Work/Data/SSD/')
%ssdb_habits( '/home/victoria.galligani/Work/Data/SSD/' )

%ssdb_init(  '/home/victoria.galligani/Work/Data/SSD/'  )


%- zenith angle
sat_alt      = 817e3;  
zenith       = 180-53;

%- reflectivity and size of the main output for GMI
%- channels = 8
nchan = length([10:1:170]); %6    [10:1:170]*1e9
nfreq = 1;
%- 400 is the maximun vertical dimension according to rt4

reflec     = 0;
arts_tb    = zeros( length(reflec),nchan );
arts_tb(:) = nan;

arts_cl    = zeros( length(reflec),nchan);
arts_cl(:) = nan;

arts_tr    = zeros( length(reflec), 400, nchan*nfreq);
arts_tr(:) = nan;

arts_tau   = zeros( length(reflec), 400, nchan*nfreq);
arts_tau(:) = nan;

%- Filenames following flag
[flag_name, graupel_habit_id]  = FlagName(flag, graupel_SSD, hail_shape);

%- Define user and paths
paths = SwitchUser(flag_name);

disp('-------------------------------')
fprintf('Hydrom type      = %s\n', flag_name);
fprintf('No. freq/channel = %d\n', nchan*nfreq*2);
disp('-------------------------------')

%- Update wfolder 
paths = UpFolder(paths,zenith);	

if clear_only == 0
	%- run arts
	%               demo_asky(pos,sat,los,flag,nfpb,ifac,cscen,solver,paths)
	solver = 'disort';
	[D,paths,R,C] = demo_asky(sat_alt,'none',zenith,flag,nfreq,solver,paths,graupel_SSD,hail_shape);
	arts_tb(1,:) = R.y;
	arts_cl(1,:) = C.y;
	arts_tr(1,1:length(D.p_grid.data),:)  = C.transm;
	arts_tau(1,1:length(D.p_grid.data),:) = C.tau;
else
	[D,paths,C] = demo_csky(sat_alt,zenith,nfreq,'none',paths);
	arts_tb(1,:) = C.y;
	arts_tr(1,1:length(D.p_grid.data),:)  = C.transm;
	arts_tau(1,1:length(D.p_grid.data),:) = C.tau;

end

%- Store the mat file
save(sprintf('%s',paths.mfolder,'/GMI_Fascod_',flag_name, '.mat'))

end
