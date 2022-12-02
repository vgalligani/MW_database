% FORMAT D = CloudScenario(D,flag,ifac,cscen,cloud.fwhm)
%
% Selection of cloud scenario 
%
% OUT   D           D  with updated bulk properties
% IN    D           Data structured
%       flag        Hydrometeor type
%       ifac        Hydrometeor content scaling factor
%       csen        Cloud scenario
%       - rect => rectangular cloud
%       - gaus => gaussian cloud
%       cloud.fwhm  Full width at half maximum
%       cloud.z     vertical cloud placement
%       cloud.bulk  bulk properties
%       - real => realistic scenario
%
% 01.01.2022 Adapted Vito Galligani
%
function D = CloudScenario(D,flag)

if nargin < 5, cloud.fwhm = 1e3;, else cloud.fwhm = fwhmi; end

if flag == 1                             % Heavy Rain only
        cloud.cdh    = D.z_field.data;
        cloud.cdh(:) = 0
        rwc_min = 0;           % kg/m3 at j=34
        rwc_median = 2.5/2000; % kg/m3 at j=17 (0km)
        rwc_max = 2.5/1000;    % kg/m3 at j=0 
        slope   = (0-D.z_field.data(17)) ./ (rwc_max-rwc_median);  %(y2-y1)/(x2-x1)
        for iz=1:34
                cloud.cdh(iz) =  (D.z_field.data(iz) - D.z_field.data(34) )/slope;
        end
        D.particle_bulkprop_field.data(1,:) = cloud.cdh

elseif flag == 2                         % Light Rain only
       print('define me')
    elseif flag == 3
       print('define me')
    end
  case 'gaus'
    if flag == 1                              % Rain Only
       cloud.cdh  = D.z_field.data(1:34);
       cloud.z    = mean(cloud.cdh);
       cloud.bulk = gauss( cloud.cdh, fwhm2si(cloud.fwhm), cloud.z );
       D.particle_bulkprop_field.data(1,1:34) = ifac*cloud.bulk;
       %
    elseif flag == 2
       print('define me')
       %
    elseif flag == 3
       print('define me')
    end
  case 'real'
    disp('A realistic scenario is not included yet.')
end

end

  
