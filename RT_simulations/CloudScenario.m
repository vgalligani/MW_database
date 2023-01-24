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
function [D, Cscat, Cmeta] = CloudScenario(D,flag, d_hail)

total_hail = 2*5; 

if flag == 1				 % Heavy Rain only
  cloud.cdh    = D.z_field.data;
  cloud.cdh(:) = 0;
  rwc_min = 0;           % kg/m3 at j=34
  rwc_median = 2.5/2000; % kg/m3 at j=17 (0km)
  rwc_max = 2.5/1000;    % kg/m3 at j=0 
  slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
  for iz=1:34
    cloud.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
  end
  D.particle_bulkprop_field.data(1,:) = cloud.cdh;
  
  C.SCAT_DATA = nan;
  C.SCAT_meta_DATA = nan;
elseif flag == 2                         % Light Rain only
  cloud.cdh    = D.z_field.data;
  cloud.cdh(:) = 0;
  rwc_min = 0;           % kg/m3 at j=34
  rwc_median = 0.25/2000; % kg/m3 at j=17 (0km)
  rwc_max = 0.25/1000;    % kg/m3 at j=0 
  slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
  for iz=1:34
    cloud.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
  end
  D.particle_bulkprop_field.data(1,:) = cloud.cdh;
  C = nan

elseif flag == 3                         % GRAUPEL 1a
  cloud_exp3        = cloud;
  cloud_exp3.cdh    = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3
.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;

elseif flag == 4                         % GRAUPEL 1b
  cloud_exp3       = cloud;
  cloud_exp3.fwhm = 500;
  cloud_exp3.cdh   = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3
.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;


elseif flag == 5                         % GRAUPEL 1c
  cloud_exp3       = cloud;
  cloud_exp3.fwhm = 2000;
  cloud_exp3.cdh   = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3
.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;

elseif flag == 6                         % HR + GRAU (EXP. A)
  cloud.cdh    = D.z_field.data;
  cloud.cdh(:) = 0;
  rwc_min = 0;           % kg/m3 at j=34
  rwc_median = 2.5/2000; % kg/m3 at j=17 (0km)
  rwc_max = 2.5/1000;    % kg/m3 at j=0 
  slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
  for iz=1:34
    cloud.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
  end
  D.particle_bulkprop_field.data(1,:) = cloud.cdh;

  cloud_exp3        = cloud;
  cloud_exp3.cdh    = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3
.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;

elseif flag == 7                         % HR + GRAU (EXP. B)
  cloud.cdh    = D.z_field.data;
  cloud.cdh(:) = 0;
  rwc_min = 0;           % kg/m3 at j=34
  rwc_median = 2.5/2000; % kg/m3 at j=17 (0km)
  rwc_max = 2.5/1000;    % kg/m3 at j=0 
  slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
  for iz=1:34
    cloud.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
  end
  D.particle_bulkprop_field.data(1,:) = cloud.cdh;

  cloud_exp3      = cloud;
  cloud_exp3.fwhm = 500;
  cloud_exp3.cdh  = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3
.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;


elseif flag == 8                         % HR + GRAU (EXP. c)
  cloud.cdh    = D.z_field.data;
  cloud.cdh(:) = 0;
  rwc_min = 0;           % kg/m3 at j=34
  rwc_median = 2.5/2000; % kg/m3 at j=17 (0km)
  rwc_max = 2.5/1000;    % kg/m3 at j=0 
  slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
  for iz=1:34
    cloud.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
  end
  D.particle_bulkprop_field.data(1,:) = cloud.cdh;

  cloud_exp3      = cloud;
  cloud_exp3.fwhm = 2000;
  cloud_exp3.cdh  = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3
.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;


elseif flag == 9                         % LR + GRAU (EXP. A)
  cloud.cdh    = D.z_field.data;
  cloud.cdh(:) = 0;
  rwc_min = 0;           % kg/m3 at j=34
  rwc_median = 0.25/2000; % kg/m3 at j=17 (0km)
  rwc_max = 0.25/1000;    % kg/m3 at j=0 
  slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
  for iz=1:34
    cloud.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
  end
  D.particle_bulkprop_field.data(1,:) = cloud.cdh;

  cloud_exp3        = cloud;
  cloud_exp3.cdh    = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3
.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;


elseif flag == 10                         % LR + GRAU (EXP. B)
  cloud.cdh    = D.z_field.data;
  cloud.cdh(:) = 0;
  rwc_min = 0;           % kg/m3 at j=34
  rwc_median = 0.25/2000; % kg/m3 at j=17 (0km)
  rwc_max = 0.25/1000;    % kg/m3 at j=0 
  slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
  for iz=1:34
    cloud.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
  end
  D.particle_bulkprop_field.data(1,:) = cloud.cdh;

  cloud_exp3      = cloud;
  cloud_exp3.fwhm = 500;
  cloud_exp3.cdh  = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3
.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;


elseif flag == 11                        % LR + GRAU (EXP. c)
  cloud.cdh    = D.z_field.data;
  cloud.cdh(:) = 0;
  rwc_min = 0;           % kg/m3 at j=34
  rwc_median = 0.25/2000; % kg/m3 at j=17 (0km)
  rwc_max = 0.25/1000;    % kg/m3 at j=0 
  slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
  for iz=1:34
    cloud.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
  end
  D.particle_bulkprop_field.data(1,:) = cloud.cdh;

  cloud_exp3      = cloud;
  cloud_exp3.fwhm = 2000;
  cloud_exp3.cdh  = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3
.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;


elseif flag == 12                       % Hail Only: simple box cloudbox of 5 levels (0-500 m) 

 %display(D.z_field.data(5)/1000)
 cloud_exp7.cdh    = D.z_field.data;
 cloud_exp7.cdh(:) = 0;
 cloud_exp7.cdh(1:5) =  (total_hail/5)/1000;

 rhohail = 500;
 d = d_hail;
 m = rhohail * 4 * pi * d^3 / 24;
 pnd = cloud_exp7.cdh/m;
 %pnd(find(pnd<1)) = 0;

 D.particle_bulkprop_field.data(1,:) = pnd;
 cloud.cdh = pnd; 

 % scat_stuff

 icerho  = 916; 
 f_grid  = [10e9:1e9:170e9];
 t_grid  = [210 240 273 283 293 303];
 rfr_index_hail = zeros( length(f_grid), length(t_grid), 1 );
 frahail = (rhohail-1) ./ (icerho-1);

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

 C.SCAT_DATA = mie_arts_scat_data( f_grid, t_grid, rfr_index_hail, 0:1:180, d/2 );
 C.SCAT_meta_DATA.version = 3;
 C.SCAT_meta_DATA.description = '';
 C.SCAT_meta_DATA.source = '';
 C.SCAT_meta_DATA.refr_index = 'Truncated Matzler';
 C.SCAT_meta_DATA.mass = nan;
 C.SCAT_meta_DATA.diameter_max = nan;
 C.SCAT_meta_DATA.diameter_volume_equ = nan;
 C.SCAT_meta_DATA.diameter_area_equ_aerodynamical = nan;


elseif flag == 13                        % Hail Only: exponential cloud. conserving total hail mas
s

 %cloud_exp7 = cloud;
 cloud_exp7.cdh    = D.z_field.data;
 cloud_exp7.cdh(:) = 0;
 hwc_min = 0;            % kg/m3 at j=18
 hwc_median = 0.25/2000; % kg/m3 at j=8 (0km)
 hwc_max = 0.4/1000;    % kg/m3 at j=0
 slope   = (0-D.z_field.data(17)) ./ (hwc_max-hwc_median);  %(y2-y1)/(x2-x1)
 for iz=1:18
   cloud_exp7.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(18) )/slope;
 end
 
 %-- OJO: keep mass total_hail
 massfix= (total_hail/1000)/sum(cloud_exp7.cdh(1:18));

 for iz=1:18
   cloud_exp7.cdh(iz) =  cloud_exp7.cdh(iz)*massfix;
 end


 rhohail = 500;
 d = d_hail;
 m = rhohail * 4 * pi * d^3 / 24;
 pnd = cloud_exp7.cdh/m;
 %pnd(find(pnd<1)) = 0;

 D.particle_bulkprop_field.data(1,:) = pnd;
 cloud.cdh = pnd;

 % scat_stuff

 icerho  = 916;
 f_grid  = [10e9:1e9:170e9];
 t_grid  = [210 240 273 283 293 303];
 rfr_index_hail = zeros( length(f_grid), length(t_grid), 1 );
 frahail = (rhohail-1) ./ (icerho-1);

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

 C.SCAT_DATA = mie_arts_scat_data( f_grid, t_grid, rfr_index_hail, 0:1:180, d/2 );
 C.SCAT_meta_DATA.version = 3;
 C.SCAT_meta_DATA.description = '';
 C.SCAT_meta_DATA.source = '';
 C.SCAT_meta_DATA.refr_index = 'Truncated Matzler';
 C.SCAT_meta_DATA.mass = nan;
 C.SCAT_meta_DATA.diameter_max = nan;
 C.SCAT_meta_DATA.diameter_volume_equ = nan;
 C.SCAT_meta_DATA.diameter_area_equ_aerodynamical = nan;

elseif flag == 14 % (like 12 but not a mono PND)

 cloud_exp7.cdh    = D.z_field.data;
 cloud_exp7.cdh(:) = 0;
 cloud_exp7.cdh(1:5) =  (total_hail/5)/1000;

 rhohail = 500;

 % Lin et al. 1983
 N0g    = 4e4;

 y_hail_int = zeros(length(cloud_exp7.cdh), 5); 

 DIAMS = [0.005 0.008 0.01 0.02 0.04];   % m
 for i = 1:length(cloud_exp7.cdh)
  lambda = ((3.14*rhohail*N0g)/(cloud_exp7.cdh(i)))^(1/3);
  for Dx = 1:length(DIAMS)
   Nhail(i,Dx) = N0g*exp(-lambda*DIAMS(Dx));
  end
   y_hail_int(i,:)= trapz(DIAMS',Nhail(i,:));
 end

 %check mass conservation:
 hdm_mass = zeros(1,length(DIAMS));
 for Dx = 1:length(DIAMS)
  hdm_mass(Dx) = rhohail * 4 * pi * DIAMS(Dx)^3 / 24;
 end

 for iz = 1:length(Nhail)
  for Dx=1:length(hdm_mass)
   x(Dx) = y_hail_int(iz,Dx)*hdm_mass(Dx);
  end
  mp_error(iz) = cloud_exp7.cdh(iz)/sum(x);
 end

 for iz = 1:length(Nhail)
  for Dx=1:length(hdm_mass)
    y_hail_int(iz,Dx) = mp_error(iz)*y_hail_int(iz,Dx);
  end
 end

 % so, hasta aca: pnd_field == y_hail_int !!! 

 %pnd = cloud_exp7.cdh/m;
 y_hail_int(isnan(y_hail_int)) = 0;
 for i = 1:length(DIAMS)
  D.particle_bulkprop_field.data(i,:) = y_hail_int(:,i);
 end
 cloud.cdh = y_hail_int;

 % scat_stuff

 icerho  = 916;
 f_grid  = [10e9:1e9:170e9];
 t_grid  = [210 240 273 283 293 303];
 rfr_index_hail = zeros( length(f_grid), length(t_grid), 1 );
 frahail = (rhohail-1) ./ (icerho-1);

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

 for i = 1:length(DIAMS)
  Cscat{i} = mie_arts_scat_data( f_grid, t_grid, rfr_index_hail, 0:1:180, DIAMS(i)/2 );
  Cmeta{i}.version = 3;
  Cmeta{i}.description = '';
  Cmeta{i}.source = '';
  Cmeta{i}.refr_index = 'Truncated Matzler';
  Cmeta{i}.mass = nan;
  Cmeta{i}.diameter_max = nan;
  Cmeta{i}.diameter_volume_equ = nan;
  Cmeta{i}.diameter_area_equ_aerodynamical = nan;
 end


if 0 
 %---- HASSTA ACA ERA EL 14. PERO EL 15 LE AGREGO LIGHT (0/25) Y AL 16 HEAVY RAIN (2.5)
  cloud.cdh    = D.z_field.data;
  cloud.cdh(:) = 0;
  rwc_min = 0;           % kg/m3 at j=34
  rwc_median = 0.25/2000; % kg/m3 at j=17 (0km)
  rwc_max = 0.25/1000;    % kg/m3 at j=0 
  slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
  for iz=1:34
    cloud.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
  end
  D.particle_bulkprop_field.data(6,:) = cloud.cdh;

end






 












end

if 0
 figure
 plot(cloud.cdh*1000, D.z_field.data/1e3, '-k','LineWidth',2);
 ylim([0 20])
 ylabel('Height (km)')
 title('Cloud profile')
 if flag == 12
   xlabel('pnd_field')
 end
end

end
