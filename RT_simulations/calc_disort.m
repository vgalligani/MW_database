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

%- RWC
%
copyfile( paths.rwc_psd, ...
          fullfile(paths.wfolder, 'cfile_psd_rwc.arts') );
%
xmlStore( fullfile(paths.wfolder, 'scat_data_rwc_file.xml'), ...
          [paths.rwc_habit,'.xml'], 'String' );
xmlStore( fullfile(paths.wfolder, 'scat_meta_rwc_file.xml'), ...
          [paths.rwc_habit,'.meta.xml'], 'String' );


%- GWC
%
copyfile( paths.gwc_psd, ...
          fullfile(paths.wfolder, 'cfile_psd_gwc.arts') );
%
xmlStore( fullfile(paths.wfolder, 'scat_data_gwc_file.xml'), ...
          [paths.gwc_habit,'.xml'], 'String' );
xmlStore( fullfile(paths.wfolder, 'scat_meta_gwc_file.xml'), ...
          [paths.gwc_habit,'.meta.xml'], 'String' );


%---- calculo aca con tmatrix hail -------------------------
%- HWC
copyfile( paths.hwc_psd, ...
          fullfile(paths.wfolder, 'hail_pnddata.xml') );
copyfile( paths.hwc_habit, ...
          fullfile(paths.wfolder, 'TestTMatrix.scat_data_single_hail.xml') );
copyfile( paths.hwc_meta_habit, ...
          fullfile(paths.wfolder, 'TestTMatrix.scat_meta_single_hail.xml') );

  
%------------------------------------------------------------

% hardcoded in .arts file

%- Run ARTS
%
%arts( sprintf('-r000 -I %s -o %s calc_disort.arts', paths.wfolder, paths.wfolder ) );
arts( sprintf('-r000 -I %s -o %s calc_disort_onlyhail.arts', paths.wfolder, paths.wfolder ) );

%- Load result
%
y     = xmlLoad( fullfile(paths.wfolder,'y.xml') );
y_geo = xmlLoad( fullfile(paths.wfolder,'y_geo.xml') );
