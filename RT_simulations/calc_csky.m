% FORMAT   [y,y_geo,T] = calc_csky(D,paths)
%
% Towards ARTS simulations
%
% OUT  y          Clear sky results
%      y_geo      Gives the Earth incident angle
%      T          Generated data structure of transmission
%
% IN   D          Generated data structure
%      paths      Structure with paths to files and folders
%
% 19.03.2020 Vasileios Barlakas and Patrick Eriksson
% 01.01.2022 Vito Galligani adopted

function [y,y_geo,T] = calc_csky(D,paths)

%- Clean work folder
%
delete( fullfile( paths.wfolder, '*.xml' ) );
delete( fullfile( paths.wfolder, '*.arts' ) );


%- Create XML files
%
xmlStoreAuto( paths.wfolder, D );

%- Set sensor specs by selecting include file
%
copyfile( paths.instrument, ...
          fullfile(paths.wfolder, 'instrument.arts') );


%- Put other include files in work folder
%
copyfile( fullfile(paths.includes,'general.arts'), ...
          fullfile(paths.wfolder, 'general.arts') );
copyfile( fullfile(paths.includes,'agendas.arts'), ...
          fullfile(paths.wfolder, 'agendas.arts') );
copyfile( fullfile(paths.includes,'agendas_surface.arts'), ...
          fullfile(paths.wfolder, 'agendas_surface.arts') );
copyfile( fullfile(paths.includes,'planet_earth.arts'), ...
          fullfile(paths.wfolder, 'planet_earth.arts') );
%
copyfile( paths.continua, ...
          fullfile(paths.wfolder, 'continua.arts') );
copyfile( fullfile(pwd,'Input','cfile_common.arts'), ...
          fullfile(paths.wfolder, 'cfile_common.arts') );

if nargout

  %- Run ARTS
  %
  arts( sprintf('-r000 -I %s -o %s calc_csky.arts', paths.wfolder, paths.wfolder ) );

  %- Load result
  %
  y          = xmlLoad( fullfile(paths.wfolder,'y.xml') );
  y_geo      = xmlLoad( fullfile(paths.wfolder,'y_geo.xml') );
  T.ppvar_p  = xmlLoad( fullfile(paths.wfolder,'ppvar_p.xml') );
  T.ppvar_t  = xmlLoad( fullfile(paths.wfolder,'ppvar_t.xml') );
  T.ppvar_vmr= xmlLoad( fullfile(paths.wfolder,'ppvar_vmr.xml') );
  T.transm   = xmlLoad( fullfile(paths.wfolder,'ppvar_trans_cumulat.xml') );
  T.tau      = xmlLoad( fullfile(paths.wfolder,'ppvar_optical_depth.xml') );
end
