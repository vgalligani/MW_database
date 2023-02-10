% FORMAT   [R,D] = demo_asky(pos,sat,los,flag,nfpb,ifac,cscen,solver,paths)
%
% Demo for creating the input for ARTS simulations for
% all-sky conditions. Initiates both clear- and all-
% sky simulations
%
% OUT  D       Generated data structure
%      R       Result data structure
%      C       Data structure for clear sky calculations
% IN   pos     Satellite altitude (a scalar)
%      sat     Satellite name
%              = none
%              = amsua
%              = mhs
%      los     Satellite zenith angle(s)
%      flag    To which hydrometeor the calculation is applied
%              = 0 => All hydrom., with cloud liquid water (clw)
%              = 1 => Rain_only (EXP1: HR) 
%              = 2 => Rain_only (EXP2: LR) 
%              = 3 => Graupel_only (EXP3: testA)
%              = 4 => 
%              = 5 => Clear sky; no clw
%      nfpb    Number of frequencies 
%      ifac    Hydrometeor content scaling factor
%      cscen   Cloud scenario
%              - rect => rectangular
%              - gaus => gaussian
%              - real => realistic
%      solver  Choose solver, i.e., rt4, disort
%      paths   Structure with paths to files and folders
%
% 19.03.2020 Patrick Eriksson, Vasileios Balrakas
%
function [D,paths,R,C] = demo_asky_bulkARTSpsd(pos,sat,los,flag,nfpb,solver,paths,graupel_Habit, total_hail) 
%
%- Conducts clear-sky simulations
%
[D,paths,C] = demo_csky( pos, los, nfpb, sat, paths );

%- Assign PSD and habit to each hydrometeor class
%
paths.rwc_psd   = fullfile( pwd, 'Input', 'psd_rwc_mp48.arts' );
paths.rwc_habit = '/home/victoria.galligani/Work/Data/ArtsScatDbase/StandardHabits/FullSet/LiquidSphere';

%paths.gwc_psd   = fullfile( pwd, 'Input', 'psd_gwc_field07t.arts' );
%paths.gwc_habit = strcat('/home/victoria.galligani/Work/Data/ArtsScatDbase/StandardHabits/FullSet/',graupel_Habit);

paths.hwc_psd   = fullfile( pwd, 'Input', 'psd_hwc_exponential.arts' );
paths.hwc_habit = fullfile( pwd, 'Input', 'hail_scatdata.xml');
paths.hwc_meta_habit = fullfile( pwd, 'Input', 'hail_metascatdata.xml');

%
D.particle_bulkprop_names.data  = {'RWC','HWC'};  
D.particle_bulkprop_names.group = 'ArrayOfString';

%
D.particle_bulkprop_field.data   = zeros( 2, length(D.p_grid.data), 1);
D.particle_bulkprop_field.group = 'Tensor4';


%- Select cloud scenario
[D] = CloudScenario_bulks(D, flag, total_hail, paths);

%- Run calculations
%
if nargout > 2
  switch solver
    case 'rt4'
      [R.y,R.y_geo] = calc_rt4( D, paths );
    case 'disort'
      [R.y,R.y_geo] = calc_disort_bulks( D, paths );
  end
end
