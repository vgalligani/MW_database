
% FORMAT   paths = UpFolder(iz)
%
% Updates the wfolder according to iz
%
% OUT   paths  Updated paths
% IN    nflag  Flagname specifying the output folder
%              depending on the type of simulation
%       paths  Structure with paths to files/folders
%       iz     Name of the sub-folder
%
% 20.05.2020 Vasileios Barlakas
%
function paths = UpFolder(paths,iz)

id             = num2str(iz);

paths.wfolder = [paths.mfolder,'/',num2str(id)];

if ~exist(paths.wfolder, 'dir')
  mkdir(paths.wfolder);
end
