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
function [D] = CloudScenario(D,flag, hail_shape)

cloud.fwhm = 1e3;

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

elseif flag == 3                         % GRAUPEL 1a
  cloud_exp3        = cloud;
  cloud_exp3.cdh    = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;

elseif flag == 4                         % GRAUPEL 1b
  cloud_exp3       = cloud;
  cloud_exp3.fwhm = 500;
  cloud_exp3.cdh   = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;


elseif flag == 5                         % GRAUPEL 1c
  cloud_exp3       = cloud;
  cloud_exp3.fwhm = 2000;
  cloud_exp3.cdh   = D.z_field.data(35:95);
  cloud_exp3.z    = mean(cloud_exp3.cdh);
  ifac_hail       = 2.0;
  cloud_exp3.bulk = zeros (size(D.z_field.data));
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3.z );
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
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3.z );
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
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3.z );
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
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3.z );
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
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3.z );
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
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3.z );
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
  cloud_exp3.bulk(35:95) = ifac_hail * gauss( cloud_exp3.cdh, fwhm2si(cloud_exp3.fwhm), cloud_exp3.z );
  D.particle_bulkprop_field.data(2,:) = cloud_exp3.bulk;
  cloud.cdh = cloud_exp3.bulk;


elseif flag == 12                       % Hail Only

 cloud_exp7 = cloud;
 cloud_exp7.cdh    = D.z_field.data;
 cloud_exp7.cdh(:) = 0;
 hwc_min = 0;            % kg/m3 at j=18
 hwc_median = 0.25/2000; % kg/m3 at j=8 (0km)
 hwc_max = 0.4/1000;    % kg/m3 at j=0
 slope   = (0-D.z_field.data(17)) ./ (hwc_max-hwc_median);  %(y2-y1)/(x2-x1)
 for iz=1:18
   cloud_exp7.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(18) )/slope;
 end

 ice_rho = 917;
 d = 1e-2;
 m = ice_rho * 4 * pi * d^3 / 24;
 pnd = cloud_exp7.cdh/m;
 %pnd(find(pnd<1)) = 0;

 D.particle_bulkprop_field.data(1,:) = pnd;
 cloud.cdh = pnd; 


elseif flag == 13                        % Hail Only BOX

 cloud_exp8 = cloud;
 cloud_exp8.cdh    = D.z_field.data;
 cloud_exp8.cdh(:) = 0;
 cloud_exp8.cdh(8:8) =  0.2;
 
 D.particle_bulkprop_field.data(3,:) = cloud_exp8.cdh;
 cloud.cdh = cloud_exp8.cdh;


end

figure
plot(cloud.cdh*1000, D.z_field.data/1e3, '-k','LineWidth',2);
ylim([0 20])
ylabel('Height (km)')
title('Cloud profile')


end
