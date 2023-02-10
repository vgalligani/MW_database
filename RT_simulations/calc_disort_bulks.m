% FORMAT   [y,y_geo] = calc_disort(D,paths)
%
% Towards ARTS simulations
%
% OUT  y          All sky results
%      y_geo      Gives the Earth incident angle
%
% IN   D          Generated data structure
%      paths      Structure with paths to files and folders
%
% 19.03.2020 Patrick Eriksson, Vasileios Balrakas
%
function [y,y_geo] = calc_disort(D,paths)

%- Common parts
%
calc_csky( D, paths );

%- HWC
copyfile( paths.hwc_psd, ...
          fullfile(paths.wfolder, 'cfile_psd_hwc.arts') );
copyfile( paths.hwc_habit, ...
          fullfile(paths.wfolder, 'hail_scatdata.xml') );
copyfile( paths.hwc_meta_habit, ...
          fullfile(paths.wfolder, 'hail_metascatdata.xml') );

% RWC:
copyfile( paths.rwc_psd, ...
          fullfile(paths.wfolder, 'cfile_psd_rwc.arts') );
xmlStore( fullfile(paths.wfolder, 'scat_data_rwc_file.xml'), ...
          [paths.rwc_habit,'.xml'], 'String' );
xmlStore( fullfile(paths.wfolder, 'scat_meta_rwc_file.xml'), ...
          [paths.rwc_habit,'.meta.xml'], 'String' );

%---
copyfile( paths.hwc_habit, ...
          fullfile(paths.wfolder, '/scat_data_hwc.xml') );
copyfile( paths.hwc_meta_habit, ...
          fullfile(paths.wfolder, '/scat_meta_data_hwc.xml') );

% hardcoded in .arts file

%- Run ARTS
arts( sprintf('-r111 -I %s -o %s calc_disort_bulks.arts', paths.wfolder, paths.wfolder ) );


%- Load result
%
y     = xmlLoad( fullfile(paths.wfolder,'y.xml') );
y_geo = xmlLoad( fullfile(paths.wfolder,'y_geo.xml') );
