% FORMAT   [D,paths,C] = demo_csky(pos,los,nfpb,sat,path[,z0,dz,ztop])
%
% OUT  D     Generated data structure
%      C     Data structure for clear sky calculations 
% IN   pos   Satellite altitude (a scalar)
%      los   Satellite zenith angle(s)
%      nfpb  Number of mono frequencies per passband
%      sat   Satellite name
%	     = none
%	     = amsua
%	     = mhs
%      path  Structure with paths to files and folders
% OPT  z0    Surface altitude. Default is 0.
%      dz    Vertical spacing. Default is 100 m. 
%      ztop  Alttude of TOA. Default is 80 km.
% 
% 19.03.2020 Vasileios Barlakas and Patrick Eriksson
%
% 01.01.2022 Adapted by Vito Galligani
function [D,paths,C] = demo_csky(pos,los,nfpb,sat,paths,z0,dz,ztop)
%
if nargin < 6, z0   = 0; end
if nargin < 7, dz   = 100; end
if nargin < 8, ztop = 80e3; end

%- Load Fascod altitudes
%
fascode = fullfile( atmlab('ARTS_XMLDATA_PATH'), 'planets', 'Earth', ...
                    'Fascod', 'midlatitude-summer', 'midlatitude-summer' );
%
F       = xmlLoad( [fascode,'.z.xml'] );

%- Create pressure grid
%
p0log10 = interp1( F.data, log10(F.grids{1}), z0 );  % ideal

% This corresponds roughly to a spacing 0f 200m up to 80 km,
% with first pressure set to match z0
zbreak  = 20e3;
dz_high = 2e3;

D.p_grid.data  = 10.^( [p0log10 : -dz/16e3 : log10(z2p_simple(zbreak)), ...
                        log10(z2p_simple(zbreak+dz_high/2)) : -dz_high/16e3 : ...
                   log10(z2p_simple(ztop)) ] )';
D.p_grid.group = 'Vector';
%
% Note that these altitudes are changed by ARTS when applying HSE
D.z_field.data  = interpp( F.grids{1}, F.data, D.p_grid.data );
D.z_field.group = 'Tensor3';
%
F               = xmlLoad( [fascode,'.t.xml'] );
D.t_field.data  = interpp( F.grids{1}, F.data, D.p_grid.data );
D.t_field.group = 'Tensor3';
%                                                        % Order between "species"
D.vmr_field.data  = zeros( 5, length(D.p_grid.data) );   % 1: N2
D.vmr_field.group = 'Tensor4';                           % 2: O2
                                                         % 3: H2O
                                                         % 4: LWC
                                                         % 5: O3
F                      = xmlLoad( [fascode,'.N2.xml'] );
D.vmr_field.data(1,:)  = interpp( F.grids{1}, F.data, D.p_grid.data )';

F                      = xmlLoad( [fascode,'.O2.xml'] );
D.vmr_field.data(2,:)  = interpp( F.grids{1}, F.data, D.p_grid.data )';

F                      = xmlLoad( [fascode,'.H2O.xml'] );
D.vmr_field.data(3,:)  = interpp( F.grids{1}, F.data, D.p_grid.data )';

F		       = xmlLoad( [fascode, '.O3.xml'] );
D.vmr_field.data(5,:)  = interpp( F.grids{1}, F.data, D.p_grid.data )';

%
% Notice that LWC exists, but it is here set 0, to stay at "clear-sky"

%- Surface properties
%
D.z_surface.data  = z0;
D.z_surface.group = 'Matrix';
%
D.surface_scalar_reflectivity.data  = 0;
D.surface_scalar_reflectivity.group = 'Vector';

%- Set line and continua files
%
D.lines_h2o.data  = fullfile( pwd, 'Input', 'abs_lines_h2o_rttov.xml' );  % !!!
D.lines_h2o.group = 'String';
%
D.lines_o3.data   = fullfile( pwd, 'Input', 'abs_lines_o3_afew_18.xml' );    % !!!
D.lines_o3.group  = 'String';
%
paths.continua   = fullfile( pwd, 'Input', 'include_mpm89_cont.arts' );

%- pos and los
%
D.sensor_pos.data  = pos;
D.sensor_pos.group = 'Matrix';
%
D.sensor_los.data  = 180;
D.sensor_los.group = 'Matrix';
%
D.antenna_dlos.data  = 180 - vec2col( los );
D.antenna_dlos.group = 'Matrix';
%
% Remove if the sensor los
D.los.data  = los;
D.los.group = 'Matrix';

%- Select instrument
%
switch sat
  case 'none'  
   paths.instrument = fullfile( pwd, 'Input', 'sensor_none.arts' );
   %
   D.f_grid.data  = 190.31e9;      % f_grid only needed for no_sensor
   D.f_grid.group = 'Vector';
  case 'amsua'
   paths.instrument = fullfile( pwd, 'Input', 'sensor_amsua.arts' );
   %
   D.met_mm_freq_number.data  = repmat({nfpb},1,15);
   %D.met_mm_freq_number.data  = {nfpb,nfpb,nfpb,nfpb,nfpb};
   D.met_mm_freq_number.group = 'ArrayOfIndex';
  case 'mhs' 
   paths.instrument = fullfile( pwd, 'Input', 'sensor_mhs.arts' );
   %
   D.met_mm_freq_number.data  = repmat({nfpb},1,5);      
   %D.met_mm_freq_number.data  = {nfpb,nfpb,nfpb,nfpb,nfpb};      
   D.met_mm_freq_number.group = 'ArrayOfIndex';
  case 'gmi'
   paths.instrument = fullfile( pwd, 'Input', 'sensor_gmi.arts' );
   %
   D.met_mm_freq_number.data  = repmat({nfpb},1,8); %10,19,22,37,89,166,183+-7, 183+-3
   %D.met_mm_freq_number.data  = {nfpb,nfpb,nfpb,nfpb,nfpb};      
   D.met_mm_freq_number.group = 'ArrayOfIndex';
 
end

%- Run calculations
%
if nargout > 2
  [C.y,C.y_geo,T] = calc_csky( D, paths );
end

%- Update data stracture for clear sky results
C.ppvar_p   = T.ppvar_p;   % pressure along the LOS
C.ppvar_t   = T.ppvar_t;   % temp along the LOS
C.ppvar_vmr = T.ppvar_vmr; % vmr along the LOS
C.transm    = T.transm;    % transmission along LOS
C.tau       = T.tau;       % tau along the LOS 
