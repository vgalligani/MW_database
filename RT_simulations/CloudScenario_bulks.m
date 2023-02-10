% FORMAT D = CloudScenario(D,flag,ifac,cscen,cloud.fwhm)
%
% Selection of cloud scenario 
%
% OUT   D           D  with updated bulk properties
% IN    D           Data structured
%       flag        Hydrometeor type
%       ifac        Hydrometeor content scaling factor
%       csen        Cloud scenario
%	- rect => rectangular cloud
%       - gaus => gaussian cloud
%       cloud.fwhm  Full width at half maximum
%       cloud.z     vertical cloud placement
%       cloud.bulk  bulk properties
%       - real => realistic scenario
%
% 01.01.2022 Adapted Vito Galligani
%
function [D] = CloudScenario(D,flag, total_hail, paths)

%total_hail = 10*5; 
rhohail = 500;

% Heavy Rain only
cloud.HR_cdh    = D.z_field.data;
cloud.HR_cdh(:) = 0;
rwc_min = 0;           % kg/m3 at j=34
rwc_median = 2.5/2000; % kg/m3 at j=17 (0km)
rwc_max = 2.5/1000;    % kg/m3 at j=0 
slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
for iz=1:34
  cloud.HR_cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
end
%D.particle_bulkprop_field.data(1,:) = cloud.HR_cdh;
  
% Light Rain only
cloud.LR_cdh    = D.z_field.data;
cloud.LR_cdh(:) = 0;
rwc_min = 0;           % kg/m3 at j=34
rwc_median = 0.25/2000; % kg/m3 at j=17 (0km)
rwc_max = 0.25/1000;    % kg/m3 at j=0 
slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
for iz=1:34
  cloud.LR_cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
end
%D.particle_bulkprop_field.data(1,:) = cloud.LR_cdh;


% Hail Only
cloud.HAIL_cdh    = D.z_field.data;
cloud.HAIL_cdh(:) = 0;
hwc_min = 0;            % kg/m3 at j=18
hwc_median = 0.25/2000; % kg/m3 at j=8 (0km)
hwc_max = 0.4/1000;    % kg/m3 at j=0
slope   = (0-D.z_field.data(17)) ./ (hwc_max-hwc_median);  %(y2-y1)/(x2-x1)
for iz=1:18
  cloud.HAIL_cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(18) )/slope;
end

% Hail scattering stuff:
%
icerho  = 916;
f_grid  = [10e9:1e9:170e9];
t_grid  = [210 240 273 283 293 303];
rfr_index_hail = zeros( length(f_grid), length(t_grid), 1 );
frahail = (rhohail-1) ./ (icerho-1);
[rfr_index_hail, rho] = get_rfr_index_hail(f_grid, t_grid, frahail);

r = [1:2:16]'*1e-3; 
%pha_mat_data = zeros( length(r), length(f_grid), length(0:1:180), 1,1,1, length(t_grid) );  pha_mat_data(:) = nan;
%ext_mat_data = zeros( length(r), length(f_grid), length(0:1:180) );  ext_mat_data(:) = nan;
%abs_vec_data = zeros( length(r), length(f_grid), length(0:1:180) );  abs_vec_data(:) = nan;
for i = 1:length(r)
	SCAT_DATA{i} = mie_arts_scat_data( f_grid, t_grid, rfr_index_hail, 0:1:180, r(i) );
	%pha_mat_data(i,:,:,:,:,:,:) = SCAT_DATA.pha_mat_data;
	%ext_mat_data(i,:,:)         = SCAT_DATA.ext_mat_data;
        %abs_vec_data(i,:,:)         = SCAT_DATA.abs_vec_data;
	SCAT_meta_DATA{i}.version = 3;
	SCAT_meta_DATA{i}.description = '';
	SCAT_meta_DATA{i}.source = '';
	SCAT_meta_DATA{i}.refr_index = 'Truncated Matzler';
	SCAT_meta_DATA{i}.mass = (rhohail * 4 * 3.14 * r(i)^3)/3;
	SCAT_meta_DATA{i}.diameter_max = 2*r(i);
	SCAT_meta_DATA{i}.diameter_volume_equ = 2*r(i);
	SCAT_meta_DATA{i}.diameter_area_equ_aerodynamical = 2*r(i);
end
% and save Scattering xml
xmlStore(paths.hwc_habit, SCAT_DATA, 'ArrayOfSingleScatteringData');
xmlStore(paths.hwc_meta_habit, SCAT_meta_DATA, 'ArrayOfScatteringMetaData');

% Define the particle_bulkprop_field for each flag escenario: 
%
if flag == 1
	D.particle_bulkprop_field.data(1,:) = cloud.HR_cdh;
elseif flag == 2
        D.particle_bulkprop_field.data(1,:) = cloud.LR_cdh;
elseif flag == 3 
        D.particle_bulkprop_field.data(1,:) = cloud.HAIL_cdh;
elseif flag == 4
        D.particle_bulkprop_field.data(1,:) = cloud.HR_cdh;
        D.particle_bulkprop_field.data(2,:) = cloud.HAIL_cdh;
elseif flag == 5
        D.particle_bulkprop_field.data(1,:) = cloud.LR_cdh;
        D.particle_bulkprop_field.data(2,:) = cloud.HAIL_cdh;
end


return

function [rfr_index_hail, rho] = get_rfr_index_hail(f_grid, t_grid, frahail)

 rfr_index_hail = zeros( length(f_grid), length(t_grid), 1 );

 % ICE
 rfr_index.ice = zeros( length(f_grid), length(t_grid) );
 for i = 1 : length(t_grid)
  rfr_index_ice(:,i) = sqrt( eps_ice_matzler06( f_grid, t_grid(i) ) ).';
 end

 % HAIL
 for i = 1 : length(t_grid)
  [rfr_index_hail(:,i) rho(:,i)] = Maxwell_Gar_mix_rule(1, ...
      rfr_index_ice(:,i), 1, 917, frahail);
 end

return 

