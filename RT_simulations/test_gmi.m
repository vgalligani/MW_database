% FORMAT   test_gmi(sat_alt,flag,nfreq,clear_only)
%
%    Simple modeling framework that creates profiles for ARTS for both
%    clear and all-sky conditions, initiates the ARTS simulations, 
%    and stores
%    any related output.
%
% OUT    *.mat      ARTS output stored in matlab format
%        fcs/fas    Files containing the input files for RTTOV simul.
% IN     sat_alt    Satellite altitude, e.g., for MHS is 817e3
%        flag       Hail/graupel/rain combination experiment selectetion
%                   Accodingly, the input/output filenames are set
%                   = 0 => All hydrom.; 
%                   = 1 => Hail only; no clw
%               !!    = 2 => Ice only; no clw
%               !!    = 3 => Rain only; no clw
%               !!    = 4 => Clear sky; no clw
%               !!    - flag_name
%               !!    - prof_filename
%       nfreq       Number of frequencies per passband of channel
%       clear_only  Flag for clear- (1) or all-sky (0) simulations.
%       ifac        Hydrometeor content scaling factor. Default = 1
%       cscen       Cloud scenario
%      	            - rect => rectangular
%                   - gaus => gaussian
%                   - real => realistic
%       solver      Choose solver, i.e., rt4, disort
%
% 19.03.2020 Vasileios Barlakas and Patrick Eriksson
% 01.01.2022 Adapted Vito Galligani
% TODO: 
%
% EXP: For clear sky run and 2 freq per passband run this as
%      follows:
%	  	 test_gmi(817e3,4,2,1)
%      For all sky run in case of a gaussian cloud 
%                test_mhs(817e3,1,1,0,1,'gaus','rt4')
%
function test_mhs(sat_alt,flag,nfreq,clear_only,ifac,cscen,solver)

%- zenith angle
zenith       = 180-53;

%- reflectivity and size of the main output for GMI
%- channels = 8
nchan = 8
%- 400 is the maximun vertical dimension according to rt4

reflec     = 0;
arts_tb    = zeros( length(reflec),nchan );
arts_tb(:) = nan;

arts_cl    = zeros( length(reflec),nchan);
arts_cl(:) = nan;

arts_tr    = zeros( length(reflec), 400, nchan*nfreq*2);
arts_tr(:) = nan;

arts_tau   = zeros( length(reflec), 400, nchan*nfreq*2);
arts_tau(:) = nan;

%- Filenames following flag
flag_name = FlagName(flag);

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
	[D,paths,R,C] = demo_asky(sat_alt,zenith,flag,nfreq,ifac,cscen,solver,paths);
	arts_tb(1,:) = R.y;
	arts_cl(1,:) = C.y;
	arts_tr(1,1:length(D.p_grid.data),:)  = C.transm;
	arts_tau(1,1:length(D.p_grid.data),:) = C.tau;
else
	%function [D,paths,C] = demo_csky(pos,los,nfpb,sat,paths,z0,dz,ztop)
	[D,paths,C] = demo_csky(sat_alt,zenith,nfreq,'gmi',paths);
	arts_tb(1,:) = C.y;
	arts_tr(1,1:length(D.p_grid.data),:)  = C.transm;
	arts_tau(1,1:length(D.p_grid.data),:) = C.tau;

end

%- Store the mat file
save(sprintf('%s',paths.mfolder,'/GMI_Fascod_',flag_name, '.mat'))

end

