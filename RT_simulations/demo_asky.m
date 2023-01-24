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
function [D,paths,R,C] = demo_asky(pos,sat,los,flag,nfpb,solver,paths,graupel_Habit,hail_d) 
%
%- Conducts clear-sky simulations
%
[D,paths,C] = demo_csky( pos, los, nfpb, sat, paths );

%- Assign PSD and habit to each hydrometeor class
%
paths.rwc_psd   = fullfile( pwd, 'Input', 'psd_rwc_mp48.arts' );
paths.rwc_habit = '/home/victoria.galligani/Work/Data/ArtsScatDbase/StandardHabits/FullSet/LiquidS
phere';
%paths.rwc_habit = fullfile(pwd, 'Input/SCAT/LiquidSpheres_TotRand', 'LiquidSpheres_TotRand');

paths.gwc_psd   = fullfile( pwd, 'Input', 'psd_gwc_field07t.arts' );
paths.gwc_habit = strcat('/home/victoria.galligani/Work/Data/ArtsScatDbase/StandardHabits/FullSet/
',graupel_Habit);

paths.hwc_psd1   = fullfile( pwd, 'Input', 'hail_pnddata1.xml' );
paths.hwc_psd2   = fullfile( pwd, 'Input', 'hail_pnddata2.xml' );
paths.hwc_psd3   = fullfile( pwd, 'Input', 'hail_pnddata3.xml' );
paths.hwc_psd4   = fullfile( pwd, 'Input', 'hail_pnddata4.xml' );
paths.hwc_psd5   = fullfile( pwd, 'Input', 'hail_pnddata5.xml' );


%--- Dummy way to include n(d)dD
paths.hwc_habit1 = fullfile( pwd, 'Input', 'TestTMatrix.scat_data_single_hail1.xml'); 
paths.hwc_meta_habit1 = fullfile( pwd, 'Input', 'TestTMatrix.scat_meta_single_hail1.xml');

paths.hwc_habit2 = fullfile( pwd, 'Input', 'TestTMatrix.scat_data_single_hail2.xml');
paths.hwc_meta_habit2 = fullfile( pwd, 'Input', 'TestTMatrix.scat_meta_single_hail2.xml');

paths.hwc_habit3 = fullfile( pwd, 'Input', 'TestTMatrix.scat_data_single_hail3.xml');
paths.hwc_meta_habit3 = fullfile( pwd, 'Input', 'TestTMatrix.scat_meta_single_hail3.xml');

paths.hwc_habit4 = fullfile( pwd, 'Input', 'TestTMatrix.scat_data_single_hail4.xml');
paths.hwc_meta_habit4 = fullfile( pwd, 'Input', 'TestTMatrix.scat_meta_single_hail4.xml');

paths.hwc_habit5 = fullfile( pwd, 'Input', 'TestTMatrix.scat_data_single_hail5.xml');
paths.hwc_meta_habit5 = fullfile( pwd, 'Input', 'TestTMatrix.scat_meta_single_hail5.xml');

%- Variables to describe hydrometeor contents
%
D.particle_bulkprop_names.data  = {'HWC','HWC','HWC','HWC','HWC'};  

%   { 'RWC','GWC','HWC'};  % , 'IWC', 'SWC' };
D.particle_bulkprop_names.group = 'ArrayOfString';
%
D.particle_bulkprop_field.data  = zeros( 5, length(D.p_grid.data), 1);
D.particle_bulkprop_field.group = 'Tensor4';

%- Select cloud scenario
%
%D = PlotCloudScenarios(D,ifac);
[D,Cscat,Cscatmeta] = CloudScenario(D,flag, hail_d);

if flag > 11
 %xmlStore(paths.hwc_psd, D.particle_bulkprop_field.data, 'Matrix');
 xmlStore(paths.hwc_psd1, D.particle_bulkprop_field.data(1,:), 'Tensor4');
 xmlStore(paths.hwc_psd2, D.particle_bulkprop_field.data(2,:), 'Tensor4');
 xmlStore(paths.hwc_psd3, D.particle_bulkprop_field.data(3,:), 'Tensor4');
 xmlStore(paths.hwc_psd4, D.particle_bulkprop_field.data(4,:), 'Tensor4');
 xmlStore(paths.hwc_psd5, D.particle_bulkprop_field.data(5,:), 'Tensor4');

 %xmlStore(paths.hwc_habit1, Cscat, 'ArrayOfSingleScatteringData');
 %xmlStore(paths.hwc_meta_habit1, Cscatmeta, 'ArrayOfScatteringMetaData');

 xmlStore(paths.hwc_habit1, Cscat{1}, 'SingleScatteringData');
 %xmlStore(paths.hwc_meta_habit1, Cscat{1}.SCAT_meta_DATA, 'ScatteringMetaData');

 if length(Cscat) > 1
 
 xmlStore(paths.hwc_habit2, Cscat{2}, 'SingleScatteringData');
 %xmlStore(paths.hwc_meta_habit2, Cscat{2}.SCAT_meta_DATA, 'ScatteringMetaData');

 xmlStore(paths.hwc_habit3, Cscat{3}, 'SingleScatteringData');
 %xmlStore(paths.hwc_meta_habit3, Cscat{3}.SCAT_meta_DATA, 'ScatteringMetaData');

 xmlStore(paths.hwc_habit4, Cscat{4}, 'SingleScatteringData');
 %xmlStore(paths.hwc_meta_habit4, Cscat{4}.SCAT_meta_DATA, 'ScatteringMetaData');

 xmlStore(paths.hwc_habit5, Cscat{5}, 'SingleScatteringData');
 %xmlStore(paths.hwc_meta_habit5, Cscat{5}.SCAT_meta_DATA, 'ScatteringMetaData');


 end

end

%A{1}.data      = D.particle_bulkprop_field.data(1,:)';
%A{1}.grids     = {D.p_grid.data, [0], [0]};
%A{1}.gridnames = {'Pressure',  'Latitude',  'Longitude'};
%xmlStore(paths.hwc_psd, A, 'ArrayOfGriddedField3');

%- Run calculations
%
if nargout > 2
  switch solver
    case 'rt4'
      [R.y,R.y_geo] = calc_rt4( D, paths );
    case 'disort'
      [R.y,R.y_geo] = calc_disort( D, paths );
  end
end
